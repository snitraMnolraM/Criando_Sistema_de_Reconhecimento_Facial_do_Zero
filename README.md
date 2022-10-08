# Criando Sistema de Reconhecimento Facial do Zero
### Como funciona o reconhecimento facial com OpenCV



O objetivo principal deste projeto é trabalhar com as bibliotecas e frameworks estudados e analisados em nossas aulas. Neste sentido, a proposta padrão envolve um sistema de detecção e reconhecimento de faces, utilizando o framework TensorFlow em conjuntos com as bibliotecas que o projetista julgue necessárias, de forma ilimitada.



![Resultado_Flamengo](https://github.com/snitraMnolraM/Criando_Sistema_de_Reconhecimento_Facial_do_Zero/blob/main/dataset/resultado/Resultado2.png?raw=true)



Antes de começar, é importante entender que a Detecção de Rosto e o Reconhecimento de Rosto são duas coisas diferentes. Na detecção de rosto, apenas o rosto de uma pessoa é detectado, o software não terá ideia de quem é essa pessoa. No Face Recognition, o software não detectará apenas o rosto, mas também reconhecerá a pessoa. Agora, deve ficar claro que precisamos realizar a detecção de rosto antes de realizar o reconhecimento de rosto.

### Reconhecimento Facial — Passo a Passo

Vamos resolver este problema passo a passo. Eu não vou explicar detalhadamente o algoritmo para evitar que o Readme fique muito extenso, mas você vai entender a ideia geral de cada um e vai aprender como criar seu próprio sistema de reconhecimento facial em Python.

- #### Passo 1: encontrar as imagens

  Usei uma extenção do Google chrome chamada Fatkun Batch Download Image, [CLICK AQUI](https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/nnjjahlikiabnchcpehcpkdeckfgnohf) para Instalar a extenção no seu Chrome.

  Você também pode usar o python, para baixar as imagens, usando o módulo **bing_image_downloader**

  ```python
  from bing_image_downloader import downloader
  ```

  Baxei em média 150 imagens de cada pessoa, separei por pasta, com nome da pessoa.  



- #### Passo 2: Extraindo as faces 

  Agora que já temos nosso banco de dados, vamos rodar o scrip para reconhecer e recortar as faces. O **extratc_faces.py** esta no repositório do GitHub.

  Módulos usados: 

  ```python
  from mtcnn import MTCNN # Implementação do detector facial MTCNN para Keras em Python
  from PIL import Image # funções para carregar imagens de arquivos e criar novas imagens
  from os import listdir # lista todos os arquivos e diretórios no diretório especificado
  from os.path import isdir # verificar se o caminho especificado é um diretório existente
  from numpy import asarray # converter a entrada em um array
  ```

  É necessario criar as pastas, com o mesmo nome de onde os arquivos serão detectados, para salvar as faces datectadas.

   

  Aumentando os dados usando o método **FLIP_LEFT_RIGHT**

  ```python
  def flip_image(image):
      img = image.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
      return img
  ```

  **Separei 10% dos dados para validação**

  `Usei o Google colab, para usar a GPU e acelerar o precesso`



- #### Passo 3: Codificar rostos

  ##### O que é uma imagem?

  Então vamos começar com o início. O que é uma imagem para um computador?

  A forma como isso é representado é simplesmente uma grade de cores.

  ![img](https://miro.medium.com/max/828/0*D8oyI8z5rjFYpqaF.png)

  ##### **Gerando os Embeddings** 

  Embeddings são uma representação do mundo, e o campo geral que os estuda é chamado de aprendizagem de representação. Existem muitas aplicações de aprendizagem de representação para imagens. Usando o Google COLAB,  o scrip **Gerando_Embeddings.ipynb**, extrai os Embeddings e salvando em CSV.

  ​

- #### Passo 4: Avaliando e criando o Modelo

  Agora que já temos o CSV com os Embeddings, podemos avaliar e escolher qual é o melhor modelo para o reconhecimento Facial.  O scrip **Avaliando_Modelos.ipynb** tem o passo-a-passo. 

  ##### Misturando os dados

  ```python
  from sklearn.utils import shuffle
  ```

  ##### Tratando das Labels

  ```python
  from sklearn.preprocessing import LabelEncoder
  ```

  #### Avaliando algoritmos

  KNN

  ```python
  from sklearn.neighbors import KNeighborsClassifier
  ```

  ```
  MODELO: KNN
  Acurácia: 94.07%
  Sensitividade: 96.9833%
  Especificidade: 92.8477%
  ```

  SVM

  ```python
  from sklearn import svm
  ```

  ```
  MODELO: SVM
  Acurácia: 98.63%
  Sensitividade: 98.0638%
  Especificidade: 99.3992%
  ```

  Keras

  ```python
  from tensorflow.keras.utils import to_categorical
  from tensorflow.keras import models, layers
  ```

  ```markdown
  MODELO: KERAS
  Acurácia: 97.83%
  Sensitividade: 97.1604%
  Especificidade: 99.0583%
  ```

  **Optei por usar o modelo Keras**

  ​

  Salvando o Modelo

  ```python
  model.save('faces.h5')
  ```





- ## Passo 5: Encontrar o nome da pessoa a partir da codificação

  Este ultimo passo é o mais fácil de todo o processo. Tudo que precisamos fazer é rodar o scrpit para encontrar a pessoa na nossa base de dados que seja mais similar à nossa imagem de teste de acordo com as medidas feitas pela rede. Agora é hora de testar nosso sistema.

  Rode o **Reconhecedor_de_faces.py** e veja o resultado.

  * A base de dados foi formada pelo 11 jogadores, do time titular do Flamengo, os demais seram classificados como Desconhecidos.

​	Após o treinamento, apliquei o classificador nas imagens. confira o resultado!



![Resultado3.png](https://github.com/snitraMnolraM/Criando_Sistema_de_Reconhecimento_Facial_do_Zero/blob/main/dataset/resultado/Resultado3.png?raw=true)

​	

![Resultado11.png](https://github.com/snitraMnolraM/Criando_Sistema_de_Reconhecimento_Facial_do_Zero/blob/main/dataset/resultado/Resultado11.png?raw=true)

​	![Resultado.png](https://github.com/snitraMnolraM/Criando_Sistema_de_Reconhecimento_Facial_do_Zero/blob/main/dataset/resultado/Resultado.png?raw=true)



![Resultado1.png](https://github.com/snitraMnolraM/Criando_Sistema_de_Reconhecimento_Facial_do_Zero/blob/main/dataset/resultado/Resultado1.png?raw=true)	



### **Conclusão**

Este tutorial abordou como usar o detector facial. Espero que você tenha entendido o artigo e conseguido implementar seu próprio sistema de reconhecimento de rosto. No entanto, existem outros métodos de detecção e reconhecimento de rosto que você pode considerar para seus projetos

Agora que você já sabe como tudo funciona, aqui estão as instruções do inicio ao fim para executar todo o processo de reconhecimento facial no seu próprio computador



[Estudo de Caso do projeto da DIO](https://academiapme-my.sharepoint.com/:w:/g/personal/kawan_dio_me/EXpal1v8435OpUIPYSvyfhsBFNMFBWnnqqiE5_o3F671DQ?rtime=5IsGlCqo2kg)

