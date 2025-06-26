import sys
import types

# Provide dummy sounddevice and soundcard modules so importing core does not fail
if 'sounddevice' not in sys.modules:
    sd_mock = types.ModuleType('sounddevice')
    def _play(*args, **kwargs):
        return None
    sd_mock.play = _play
    sys.modules['sounddevice'] = sd_mock

if 'soundcard' not in sys.modules:
    sc_mock = types.ModuleType('soundcard')
    sys.modules['soundcard'] = sc_mock
