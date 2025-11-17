import chromadb
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException,  WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from smartscan.constants import DB_DIR, SMARTSCAN_CONFIG_PATH, MODEL_REGISTRY
from smartscan.config import load_config
from smartscan.utils.embeddings import get_image_encoder, get_text_encoder
from smartscan.utils.file import are_valid_files, get_files_from_dirs
from smartscan.indexer.indexer import FileIndexer
from smartscan.indexer.indexer_listener import FileIndexerWebSocketListener

config = load_config(SMARTSCAN_CONFIG_PATH)

client = chromadb.PersistentClient(path=DB_DIR)
text_store = client.get_or_create_collection(
    name=f"{config.text_encoder_model}_text_collection",
    metadata={"description": "Collection for text documents"}
)
image_store = client.get_or_create_collection(
    name=f"{config.image_encoder_model}_image_collection",
    metadata={"description": "Collection for images"}
) 
video_store = client.get_or_create_collection(
    name=f"{config.image_encoder_model}_video_collection",
    metadata={"description": "Collection for videos"}
)

image_encoder_path = MODEL_REGISTRY[config.image_encoder_model]['path']
image_encoder = get_image_encoder(image_encoder_path)

text_encoder_path = MODEL_REGISTRY[config.text_encoder_model]['path']
text_encoder = get_text_encoder(text_encoder_path)

image_encoder.init()
text_encoder.init()


MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXT = ('png', 'jpg', 'jpeg', 'bmp', 'webp')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["*"],
    max_age=3600,
)

@app.post("/api/search/image")
async def search_images(query_image: UploadFile = File(...),threshold: float = Form(0.6),):
    if query_image.filename is None:
        raise HTTPException(status_code=400, detail="Missing query_image")
    if not are_valid_files(ALLOWED_EXT, [query_image.filename]):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        image = Image.open(query_image.file)
        query_embedding = await run_in_threadpool(image_encoder.embed, image)

    except Exception as _:
            raise HTTPException(status_code=500, detail="Error generating embedding")

    try:
          results = image_store.query(query_embeddings=[query_embedding])
    except Exception as _:
            raise HTTPException(status_code=500, detail="Error querying database")

    ids = []
    for id_, distance in zip(results['ids'][0], results['distances'][0]):
        if distance <= threshold:
            ids.append(id_)
    
    return JSONResponse({"results": ids})


@app.post("/api/search/video")
async def search_videos(query_image: UploadFile = File(...),threshold: float = Form(0.6),):
    if query_image.filename is None:
        raise HTTPException(status_code=400, detail="Missing query_image")
    if not are_valid_files(ALLOWED_EXT, [query_image.filename]):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        image = Image.open(query_image.file)
        query_embedding = await run_in_threadpool(image_encoder.embed, image)

    except Exception as _:
            raise HTTPException(status_code=500, detail="Error generating embedding")

    try:
          results = video_store.query(query_embeddings=[query_embedding])
    except Exception as _:
            raise HTTPException(status_code=500, detail="Error querying database")

    ids = []
    for id_, distance in zip(results['ids'][0], results['distances'][0]):
        if distance <= threshold:
            ids.append(id_)
    
    return JSONResponse({"results": ids})


@app.post("/api/search/text")
async def search_documents(query: str, threshold: float = Form(0.6),):
    if query is None:
        raise HTTPException(status_code=400, detail="Missing query text")
  
    try:
        query_embedding = await run_in_threadpool(text_encoder.embed, query)

    except Exception as _:
            raise HTTPException(status_code=500, detail="Error generating embedding")

    try:
          results = text_store.query(query_embeddings=[query_embedding])
    except Exception as _:
            raise HTTPException(status_code=500, detail="Error querying database")

    ids = []
    for id_, distance in zip(results['ids'][0], results['distances'][0]):
        if distance <= threshold:
            ids.append(id_)
    
    return JSONResponse({"results": ids})


@app.websocket("/ws/index")
async def index(ws: WebSocket):
    await ws.accept()

    listener = FileIndexerWebSocketListener(ws)
    indexer = FileIndexer(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        image_store=image_store,
        text_store=text_store,
        video_store=video_store,
        listener=listener,
    )

    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("action") == "index":
                dirpaths = msg.get("dirs", [])
                allowed_exts = indexer.valid_img_exts + indexer.valid_txt_exts + indexer.valid_vid_exts
                files = get_files_from_dirs(dirpaths, allowed_exts=allowed_exts)
                await indexer.run(files)
            elif msg.get("action") == "stop":
                break
    except WebSocketDisconnect:
        print("Client disconnected")
