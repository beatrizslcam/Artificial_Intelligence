import glob
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 


def salva_imagem_com_predicao(imagem, img_filename, predicao):
  img = Image.fromarray(imagem[:,:,:])

  draw = ImageDraw.Draw(img)
  font = ImageFont.truetype("FreeMono.ttf", 11)
  fillcolor = "black"
  shadowcolor = "white"
  x, y = 1, 1

  text = "{:.3f}".format(predicao)

  # thin border
  draw.text((x-1, y), text, font=font, fill=shadowcolor)
  draw.text((x+1, y), text, font=font, fill=shadowcolor)
  draw.text((x, y-1), text, font=font, fill=shadowcolor)
  draw.text((x, y+1), text, font=font, fill=shadowcolor)

  # thicker border
  draw.text((x-1, y-1), text, font=font, fill=shadowcolor)
  draw.text((x+1, y-1), text, font=font, fill=shadowcolor)
  draw.text((x-1, y+1), text, font=font, fill=shadowcolor)
  draw.text((x+1, y+1), text, font=font, fill=shadowcolor)

  draw.text((x, y), text, font=font, fill=fillcolor)

  img.save(img_filename)
  return 
