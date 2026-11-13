import sys,os
import numpy as np
import xml.etree.ElementTree as ET
from lxml import etree
import mujoco as mj
import platform
import shutil
import subprocess
import pygments
from IPython.display import HTML, display
from copy import deepcopy
import colorsys
sys.path.append('../../package/kinematics_helper/')
sys.path.append('../../package/mujoco_helper/')
sys.path.append('../../package/utility/')
sys.path.append('../../asset/')
from mujoco_parser import *
from utils import *

# Render the MuJoCo viewer on the NVIDIA GPU (PRIME offload), only when an NVIDIA driver is
# present; values already set in the environment are kept.
if platform.system() == "Linux" and shutil.which("nvidia-smi"):
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__VK_LAYER_NV_optimus", "NVIDIA_only")

def is_macos_dark_theme():
    """Return True if macOS is in dark mode."""
    if sys.platform == 'darwin':
        try:
            result = subprocess.run(
                ['defaults', 'read', '-g', 'AppleInterfaceStyle'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return "Dark" in result.stdout
        except Exception:
            return False
    return False

def is_linux_dark_theme():
    """Return True if the GNOME GTK theme is a dark one."""
    if sys.platform.startswith('linux'):
        try:
            result = subprocess.run(
                ['gsettings', 'get', 'org.gnome.desktop.interface', 'gtk-theme'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            theme = result.stdout.strip().strip("'")
            return "dark" in theme.lower()
        except Exception:
            return False
    return False

def is_dark_theme():
    """Return True if the desktop uses a dark theme (macOS or GNOME; False elsewhere)."""
    import sys
    if sys.platform == 'darwin':
        return is_macos_dark_theme()
    elif sys.platform.startswith('linux'):
        return is_linux_dark_theme()
    else:
        return False
    
def print_xml(xml_input,color=True):
    """Pretty-print an XML string or Element: syntax-highlighted HTML in Jupyter if color, plain text otherwise."""
    if isinstance(xml_input, ET.Element):
        rough_string = ET.tostring(xml_input, encoding='unicode')
    else:
        rough_string = xml_input
    if color: 
        print_style = 'monokai' if is_dark_theme() else 'lovelace'
        formatter = pygments.formatters.HtmlFormatter(style=print_style)
        lexer = pygments.lexers.XmlLexer()
        highlighted = pygments.highlight(rough_string, lexer, formatter)
        display(HTML(f"<style>{formatter.get_style_defs()}</style>{highlighted}"))
    else:
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.fromstring(rough_string, parser=parser)
        pretty_xml = etree.tostring(tree, pretty_print=True, encoding='unicode')
        print(pretty_xml)

def get_rainbow_color(i, n):
        """
        i: current index (0 <= i < n)
        n: total count
        """
        # spread Hue over 0 ~ 1 in HSV space
        h = i / n
        s = 1.0    # saturation
        v = 1.0    # value (brightness)

        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return [r, g, b, 1.0]  # RGBA (alpha=1.0)