# Paladium CLI

A **Paladium CLI** é uma ferramenta de linha de comando para detecção de veículos e placas, que permite processar imagens, vídeos e até URLs do YouTube, além de pastas inteiras contendo esses arquivos. Ela utiliza modelos baseados em YOLOv11 para detecção de veículos, [YOLOv5](https://github.com/keremberke/awesome-yolov5-models) para detecção de placas e um modelo OCR ([ONNXPlateRecognizer](https://github.com/ankandrew/fast-plate-ocr)) para leitura das placas. A ferramenta gera um arquivo CSV estruturado com os resultados e, quando ativado, exibe em modo debug as imagens/vídeos com as caixas delimitadoras e os textos inferidos. 

## Funcionalidades

### Processamento de Vídeo
- Processa vídeos locais ou URLs do YouTube
- Realiza detecção de veículos e placas
- Realiza OCR para leitura das placas
- Pode salvar um vídeo anotado
- Gera um CSV com os resultados (número do frame, ID do veículo, bounding boxes, textos de placas e scores)

### Processamento de Imagem
- Processa uma única imagem para detecção de veículos e placas
- Exibe os resultados (placas detectadas e suas confianças) no terminal
- Exibe a imagem anotada em modo debug, aguardando interação do usuário

### Processamento de Pasta (Recursivo)
- Percorre uma pasta e todas as suas subpastas para identificar arquivos de vídeo e imagem
- Processa cada arquivo individualmente
- Agrega os resultados em um único arquivo CSV, incluindo o caminho relativo do arquivo
- Em modo debug, exibe as imagens ou o vídeo em tempo real; para imagens, aguarda a tecla **c** para avançar para o próximo item

### Modo Debug
- Permite visualizar, em tempo real, as inferências com as caixas delimitadoras e os textos (placas)
- Para vídeos: exibição ao vivo durante o processamento
- Para imagens: exibição da imagem anotada com pausa até o usuário pressionar a tecla **c** para continuar (em caso de imagens), ou **q** para interromper um vídeo 

### Integração com Modelos
- **Detecção de Veículos:** Utiliza um modelo YOLOv11 exportado para ONNX
- **Detecção de Placas:** Utiliza um modelo baseado em YOLOv5
- **OCR para Placas:** Utiliza o ONNXPlateRecognizer para leitura das placas detectadas

## Estrutura do Projeto

```
./
└── paladium
    ├── cli.py                 # Arquivo principal da CLI com os comandos (video, image, folder)
    ├── generate_model.py      # Script para exportar o modelo YOLO para ONNX
    ├── image_processor.py     # Processamento e anotação de imagens
    ├── video_processor.py     # Processamento e anotação de vídeos
    └── utils
        ├── __init__.py
        ├── file_io.py         # Função para escrita do arquivo CSV com os resultados
        ├── ocr.py             # Implementação do OCR otimizado para placas
        ├── plate_data.py      # Estrutura de dados para representar uma placa
        ├── preprocessing.py   # Pré-processamento para melhoria do OCR
        ├── tracking.py        # Implementação do tracker para associar detecções de veículos
        └── vehicle_data.py    # Estrutura de dados para representar um veículo
```

## Requisitos

- Python 3.11+
- OpenCV
- Ultralytics (YOLO)
- yolov5
- PyTorch
- yt-dlp
- fast_plate_ocr (ou similar, conforme a implementação do OCR)
- Outras dependências que podem ser instaladas via `uv` (consulte o arquivo `pyproject.toml`, se houver)


## Instalação

1. **Instale uv (ou seu framework favorito)**:
```bash
pip install uv
```

2. **Crie e ative um ambiente virtual com [uv](https://docs.astral.sh/uv/) (opcional, mas recomendado):**

```bash
uv sync
```

3. **Exporte o modelo YOLO para ONNX (se necessário):**

```bash
uv run python generate_model.py
```

## Uso

A CLI possui três comandos principais:

### 1. Processar Vídeo

Processa um vídeo local ou uma URL do YouTube para detecção de placas.

**Exemplo:**

```bash
uv run python cli.py video --input path/to/video.mp4 --output resultados.csv --frame-skip 2 --debug
```

- `--input`: Caminho do arquivo de vídeo ou URL do YouTube
- `--output`: Caminho para salvar o CSV com os resultados
- `--frame-skip`: Número de frames a pular (para acelerar o processamento)
- `--video-output` (opcional): Caminho para salvar o vídeo anotado
- `--debug`: Ativa o modo debug, exibindo os frames com as inferências em tempo real

### 2. Processar Imagem

Processa uma única imagem para detecção de placas.

**Exemplo:**

```bash
uv run python cli.py image --input path/to/image.jpg --debug
```

- `--input`: Caminho para a imagem
- `--debug`: (Opcional) Exibe a imagem anotada em modo debug, aguardando que o usuário pressione **c** para continuar

### 3. Processar Pasta

Processa recursivamente todos os vídeos e imagens dentro de uma pasta e suas subpastas, agregando os resultados em um único arquivo CSV.

**Exemplo:**

```bash
uv run python cli.py folder --input-folder path/to/folder --output aggregated_results.csv --frame-skip 2 --debug
```

- `--input`: Caminho da pasta contendo os arquivos
- `--output`: Caminho para salvar o CSV agregado com os resultados
- `--frame-skip`: Número de frames a pular durante o processamento de vídeos
- `--debug`: Ativa o modo debug:
  - Para **vídeos**: os frames são exibidos ao vivo durante o processamento
  - Para **imagens**: a imagem anotada é exibida e aguarda que o usuário pressione a tecla **c** para prosseguir

## Debug e Interação

### Vídeos
No modo debug, o vídeo é exibido ao vivo (com `cv2.imshow` e `cv2.waitKey(1)`) dentro do método `process_video`. Assim, é possível visualizar o andamento do processamento sem interrupção (exceto se apertando a tecla **q**).

### Imagens
No modo debug, a imagem anotada é exibida e o programa aguarda até que o usuário pressione a tecla **c** para fechar a janela e continuar.

## Fontes interessantes

- [Dataset de placas brasileiras](https://github.com/raysonlaroca/ufpr-alpr-dataset?tab=readme-ov-file)