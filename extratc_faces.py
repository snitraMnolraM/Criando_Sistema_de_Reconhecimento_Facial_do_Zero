from mtcnn import MTCNN
from PIL import Image
from os import listdir
from os.path import isdir
from numpy import asarray

detector = MTCNN()

def extrair_face(arquivo,size=(160,160)):

    img = Image.open(arquivo) # caminho do arquivo
    img = img.convert('RGB')
    array = asarray(img)
    results = detector.detect_faces(array)

    x1, y1, width, height = results[0]['box']
    x2, y2  = x1 + width, y1 + height
    face = array[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(size)


    return image

def flip_image(image):
    img = image.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
    return img




def load_images(directory_src, directory_target):
    index = 0

    for filename in listdir(directory_src):
        path = (f'{directory_src}{filename}')
        path_tg = (f'{directory_target}')
        path_tg_flip = directory_target + str(index) + '.png'

        try:
            face = extrair_face(path)
            flip = flip_image(face)
            face.save(f'{path_tg}/{index}.jpg', quality=100, optimize=True, progressive=True)
            flip.save(path_tg_flip, quality=100, optimize=True, progressive=True)

            index += 1
        except:
            print('Erro na Imagem {}'.format(path))

    #print(directory_src)
    #print(directory_target)


def load_dir(directory_src, directory_target):
    for subdir in listdir(directory_src):

        path = directory_src + subdir + "/"

        path_tg = directory_target + subdir + "/"

        if not isdir(path):
            continue

        load_images(path, path_tg)


if __name__ == "__main__":
    load_dir("/Users/cabal/Documents/Flamengo/Flamengo/","/Users/cabal/Documents/Flamengo/Faces_flip/")