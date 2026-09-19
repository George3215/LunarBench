"""Resolve MJCF file references before compiling a copied collision window."""
from pathlib import Path
import xml.etree.ElementTree as ET


def scene_tree(path):
    path = Path(path).resolve()
    tree = ET.parse(path)
    root = tree.getroot()
    compiler = root.find('compiler')
    for attribute in ('meshdir', 'texturedir'):
        if compiler is not None and compiler.get(attribute):
            compiler.set(attribute, str((path.parent / compiler.get(attribute)).resolve()))
    for element in root.findall('asset/hfield'):
        if element.get('file'):
            element.set('file', str((path.parent / element.get('file')).resolve()))
    return tree
