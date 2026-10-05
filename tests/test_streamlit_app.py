import importlib.util
from pathlib import Path
import unittest


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "Streamlit is optional")
class StreamlitViewTests(unittest.TestCase):
    def test_all_three_views_render_without_exceptions(self) -> None:
        from streamlit.testing.v1 import AppTest

        script = Path(__file__).resolve().parents[1] / "streamlit_app.py"
        app = AppTest.from_file(str(script), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertIn("Flow prediction", [header.value for header in app.header])
        for view, header in (
            ("Poisoning / quarantine", "Poisoning and quarantine review"),
            ("Continual comparison", "Continual comparison — saved experiment playback"),
        ):
            app.sidebar.radio[0].set_value(view)
            app.run(timeout=20)
            self.assertFalse(app.exception)
            self.assertIn(header, [item.value for item in app.header])


if __name__ == "__main__":
    unittest.main()
