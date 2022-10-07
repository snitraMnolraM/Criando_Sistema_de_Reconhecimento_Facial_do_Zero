from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model


pessoas = ["Arrascaeta 14", "David Luiz 23", "Desconhecido", "Everton Ribeiro 7",
     "Filipe Luis 16", "Gabigol 9", "Joao Gomes 35", "Leo Pereira 4",
     "Pedro 21", "Rodinei 22", "Santos 20", "Thiago Maia 8"]
num_pessoas = len(pessoas)

cap = cv2.VideoCapture('dataset/Img_Teste/fla2.webp')
detector = MTCNN()
facenet = load_model('Modelos/facenet_keras.h5', compile=False)
model = load_model("Modelos/faces.h5", compile=False)

def extrair_face(image, box, required_size=(160,160)):

    pixels = np.asarray(image)

    x1, y1, width, height = box

    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(required_size)
    return np.asarray(image)


def get_embedding(facenet, face_pixels):

    face_pixels = face_pixels.astype('float32')

    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean)/ std

    samples = np.expand_dims(face_pixels, axis=0)

    yhat = facenet.predict(samples)
    return yhat[0]


while (cap.isOpened()):

    _, frame = cap.read()

    faces = detector.detect_faces(frame)

    for face in faces:
        confidence = face['confidence']*100

        if confidence>=98:
            x1, y1, w, h = face['box']
            face = extrair_face(frame, face['box'])

            face = face.astype('float32')/255
            emb = get_embedding(facenet, face)
            tensor = np.expand_dims(emb, axis=0)
            classe = model.predict_classes(tensor)[0]
            prob = model.predict_proba(tensor)
            prob = prob[0][classe]*100

            user = str(pessoas[classe]).upper()

            color = (0, 255, 199)

            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), color, 2)

            font = cv2.FONT_HERSHEY_TRIPLEX
            font_scale = 0.6

            cv2.putText(frame, user, (x1, y1-10), font, fontScale=font_scale, color=color,
                        thickness=1)
    cv2.imshow("RECONHECIMENTO DE FACES", frame)


    key = cv2.waitKey(0)

    if key == ord("s"):  # pressionado "s" para salva a imagem
        cv2.imwrite("dataset/resultado/Resultado11.png", frame)
    if key==27:#ESC
        break

cap.release()
cv2.destroyAllWindows()