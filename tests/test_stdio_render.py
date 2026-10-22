"""Opt-in real STDIO MCP -> LMMS smoke test, with no model or network calls.

Prepare an isolated environment and LMMS_TEST_EXECUTABLE separately. Set
LMMS_TEST_STDIO_RENDER=1 to enable. Nothing is downloaded, installed or extracted
by this test. Artifacts stay under pytest's temporary directory by default.
"""

import array
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import time
import wave
from xml.etree import ElementTree as ET

import anyio
import pytest

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


@pytest.mark.skipif(
    os.environ.get('LMMS_TEST_STDIO_RENDER') != '1',
    reason='Set LMMS_TEST_STDIO_RENDER=1 in a prepared isolated environment',
)
def test_stdio_roundtrip_and_real_render(tmp_path):
    executable = os.environ.get('LMMS_TEST_EXECUTABLE')
    assert executable, 'LMMS_TEST_EXECUTABLE must explicitly select the test application'
    assert Path(executable).is_file() and os.access(executable, os.X_OK)
    output = Path(os.environ.get('LMMS_TEST_ARTIFACT_DIR', tmp_path / 'stdio-render'))
    output.mkdir(parents=True, exist_ok=False)
    project = output / 'Pulso-MCP.mmp'
    audio = output / 'Pulso-MCP.wav'
    transcript = []
    report = {'status': 'FAIL', 'transport': 'stdio', 'sdk_version': version('mcp')}
    env = {
        'PYTHONPATH': os.pathsep.join(filter(None, (
            str(Path(__file__).resolve().parents[1] / 'src'),
            os.environ.get('PYTHONPATH'),
        ))),
        'PYTHONDONTWRITEBYTECODE': '1',
        'LMMS_EXECUTABLE': executable,
        'LMMS_PROJECTS_DIR': str(output),
        'QT_QPA_PLATFORM': 'offscreen',
        'OMP_NUM_THREADS': '1',
    }
    # Explicit Qt/plugin selections may be needed by the prepared installation.
    for key in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'LMMS_PLUGIN_DIR'):
        if key in os.environ:
            env[key] = os.environ[key]
    parameters = StdioServerParameters(
        command=sys.executable, args=['-m', 'lmms_mcp'], env=env, cwd=str(output),
    )

    async def request(session, name, arguments=None, *, expect_error=False, parse=True):
        started = time.monotonic()
        result = await session.call_tool(name, arguments or {}, read_timeout_seconds=45)
        transcript.append({
            'name': name, 'arguments': arguments or {},
            'elapsed_seconds': round(time.monotonic() - started, 6),
            'response': result.model_dump(mode='json', by_alias=True),
        })
        assert bool(result.is_error) is expect_error, result
        text = '\n'.join(block.text for block in result.content if block.type == 'text')
        return json.loads(text) if parse else text

    async def exercise():
        with anyio.fail_after(90):
            with (output / 'server-stderr.log').open('w') as errors:
                async with stdio_client(parameters, errlog=errors) as streams:
                    async with ClientSession(*streams, read_timeout_seconds=20) as session:
                        initialized = await session.initialize()
                        report['protocol_version'] = initialized.protocol_version
                        report['server_info'] = initialized.server_info.model_dump(mode='json')
                        listing = await session.list_tools()
                        assert listing.next_cursor is None, 'Review pagination before widening this smoke test'
                        names = {tool.name for tool in listing.tools}
                        required = {
                            'create_project', 'add_instrument_track', 'add_notes_batch',
                            'save_project', 'load_project', 'get_project_info',
                            'get_lmms_info', 'render_project', 'set_tempo',
                        }
                        assert required <= names
                        report['tool_count'] = len(names)
                        report['tool_names'] = sorted(names)
                        (output / 'catalog.json').write_text(listing.model_dump_json(by_alias=True))

                        installation = await request(session, 'get_lmms_info')
                        assert installation['found'] is True
                        assert installation['version'] is not None
                        expected = os.environ.get('LMMS_TEST_EXPECTED_VERSION')
                        if expected:
                            assert installation['version'] == expected
                        report['lmms_version'] = installation['version']
                        assert 'tripleoscillator' in installation['instruments_verified_available']

                        # SDK-level validation error, followed by a successful operation.
                        await request(session, 'set_tempo', {'bpm': 'not-an-integer'},
                                      expect_error=True, parse=False)
                        await request(session, 'create_project', {'bpm': 108, 'master_volume': 50})
                        # An application error inside a successful MCP envelope is NOT success.
                        invalid = await request(session, 'add_instrument_track', {
                            'name': 'Invalid control', 'instrument': 'not_a_real_instrument_742',
                        })
                        assert 'error' in invalid
                        info = await request(session, 'get_project_info')
                        assert info['track_count'] == 0

                        lead = await request(session, 'add_instrument_track', {
                            'name': 'Pulso MCP - melodia', 'instrument': 'tripleoscillator',
                            'volume': 18, 'panning': 10,
                        })
                        bass = await request(session, 'add_instrument_track', {
                            'name': 'Pulso MCP - bajo', 'instrument': 'tripleoscillator',
                            'volume': 14, 'panning': -10,
                        })
                        assert lead['track_index'] == 0 and bass['track_index'] == 1
                        # Native LMMS key indices. End exactly at the second bar boundary.
                        melody = [60, 64, 67, 64, 69, 67, 64, 60]
                        lead_notes = [
                            {'key': key, 'pos': index * 48, 'length': 40, 'volume': 70}
                            for index, key in enumerate(melody)
                        ]
                        bass_notes = [
                            {'key': key, 'pos': index * 96, 'length': 96, 'volume': 65}
                            for index, key in enumerate((24, 31, 21, 28))
                        ]
                        for index, notes in ((0, lead_notes), (1, bass_notes)):
                            added = await request(session, 'add_notes_batch', {
                                'track_index': index, 'notes': notes,
                            })
                            assert added['added'] == len(notes)
                            assert all('error' not in item for item in added['notes'])
                        await request(session, 'save_project', {'path': str(project), 'compressed': False})
                        # Replace in-memory state before reloading: a no-op load cannot pass.
                        await request(session, 'create_project', {'bpm': 120})
                        empty = await request(session, 'get_project_info')
                        assert empty['bpm'] == 120 and empty['track_count'] == 0
                        await request(session, 'load_project', {'path': str(project)})
                        restored = await request(session, 'get_project_info')
                        assert restored['bpm'] == 108 and restored['track_count'] == 2
                        restored_notes = sum(
                            len(pattern['notes']) for track in restored['tracks']
                            for pattern in track['patterns']
                        )
                        assert restored_notes == 12
                        disk = ET.parse(project).getroot()
                        assert len(disk.findall('.//note')) == 12
                        rendered = await request(session, 'render_project', {
                            'output_path': str(audio), 'file_format': 'wav', 'samplerate': 44100,
                        })
                        assert rendered.get('success') is True and rendered['returncode'] == 0
                        report['render'] = rendered
            report['session_closed'] = True

    try:
        anyio.run(exercise)
        with wave.open(str(audio), 'rb') as wav:
            assert wav.getsampwidth() == 2
            channels, rate, frames = wav.getnchannels(), wav.getframerate(), wav.getnframes()
            pcm = array.array('h', wav.readframes(frames))
        if sys.byteorder != 'little':
            pcm.byteswap()
        assert channels == 2 and rate == 44100 and frames > 0
        peak = max(abs(value) for value in pcm)
        rms = math.sqrt(sum(value * value for value in pcm) / len(pcm))
        clipping = sum(abs(value) >= 32767 for value in pcm)
        assert 4 <= frames / rate <= 9 and rms > 10 and clipping == 0
        report['audio'] = {
            'channels': channels, 'sample_rate_hz': rate, 'sample_width_bytes': 2,
            'duration_seconds': round(frames / rate, 4),
            'peak_dbfs': round(20 * math.log10(peak / 32768), 3),
            'rms_dbfs': round(20 * math.log10(rms / 32768), 3),
            'full_scale_samples': clipping,
        }
        report['tool_calls'] = len(transcript)
        report['controls'] = {
            'invalid_argument_is_error': True,
            'application_error_inspected_inside_content': True,
            'invalid_instrument_did_not_add_track': True,
            'reload_replaced_different_in_memory_project': True,
            'xml_reopened_independently': True,
        }
        report['listening_review'] = 'NOT_PERFORMED'
        report['status'] = 'PASS'
    finally:
        (output / 'transcript.json').write_text(json.dumps(transcript, indent=2))
        (output / 'smoke.json').write_text(json.dumps(report, indent=2))
