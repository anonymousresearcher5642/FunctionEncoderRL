# run this python code to download the things

import urllib.request
urllib.request.urlretrieve('http://www.atarimania.com/roms/Roms.rar','Roms.rar')

### run these commands in terminal to install the roms
# !sudo apt-get install unrar
# !unrar x Roms.rar
# !mkdir rars
# !mv HC\ ROMS.zip   rars
# !mv ROMS.zip  rars
# !python -m atari_py.import_roms rars