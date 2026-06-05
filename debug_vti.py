#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import base64
import struct
import numpy as np

filepath = '/home/franek/TCLB/output/01clean_canal_VTK_P00_00000500.vti'

tree = ET.parse(filepath)
root = tree.getroot()
print(f"Root tag: {root.tag}")

# Find ImageData element
imgdata = root.find('ImageData')
print(f"ImageData: {imgdata}")

if imgdata:
    print(f"WholeExtent: {imgdata.get('WholeExtent')}")
    extent = list(map(int, imgdata.get('WholeExtent', '0 0 0 0 0 0').split()))
    nx = extent[1] - extent[0] + 1
    ny = extent[3] - extent[2] + 1
    nz = extent[5] - extent[4] + 1
    print(f"Dimensions: nx={nx}, ny={ny}, nz={nz}")
    
    piece = imgdata.find('Piece')
    print(f"Piece: {piece}")
    
    if piece:
        celldata = piece.find('CellData')
        print(f"CellData: {celldata}")
        
        if celldata:
            for dataarray in celldata.findall('DataArray'):
                name = dataarray.get('Name')
                dtype = dataarray.get('type')
                fmt = dataarray.get('format')
                enc = dataarray.get('encoding')
                print(f"DataArray: name={name}, type={dtype}, format={fmt}, encoding={enc}")
                
                if name == 'Rho':
                    # Get text
                    text = dataarray.text
                    if text:
                        text_clean = ''.join(text.split())
                        print(f"Text length (raw): {len(dataarray.text)}")
                        print(f"Text length (cleaned): {len(text_clean)}")
                        print(f"First 100 chars: {text_clean[:100]}")
                        
                        # Try to decode
                        try:
                            decoded = base64.b64decode(text_clean)
                            print(f"Decoded length: {len(decoded)} bytes")
                            print(f"Number of doubles: {len(decoded) // 8}")
                            
                            # Try to unpack first few
                            num_vals = min(10, len(decoded) // 8)
                            values = struct.unpack(f'{num_vals}d', decoded[:num_vals*8])
                            print(f"First {num_vals} values: {values}")
                        except Exception as e:
                            print(f"Decode error: {e}")
