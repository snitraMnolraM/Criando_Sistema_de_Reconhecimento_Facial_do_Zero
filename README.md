# Criando Sistema de Reconhecimento Facial do Zero
### Como funciona o reconhecimento facial com OpenCV

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

>    **Usei o Google colab, para usar a GPU e acelerar o precesso**



- #### Passo 3: Codificar rostos

  ##### O que é uma imagem?

  Então vamos começar com o início. O que é uma imagem para um computador?

  A forma como isso é representado é simplesmente uma grade de cores.

  ![img](https://miro.medium.com/max/828/0*D8oyI8z5rjFYpqaF.png)

  ##### **Gerando os Embeddings** 

  Embeddings são uma representação do mundo, e o campo geral que os estuda é chamado de aprendizagem de representação. Existem muitas aplicações de aprendizagem de representação para imagens. Usando o Google COLAB,  o scrip **Gerando_Embeddings.ipynb**, extrai os Embeddings e salvando em CSV.

  ​

- #### Passo 4: Avaliando e criando o Modelo

   

  ```markdown
    MODELO: KERAS
    Acurácia: 97.83%
    Sensitividade: 97.1604%
    Especificidade: 99.0583%
  ```

  ​



​	



​	