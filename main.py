####################################### IMPORT #################################
import json
import pandas as pd
from PIL import Image
from loguru import logger
import uuid
import sys
import torch
import torchvision.transforms as transforms
from fuzzywuzzy import fuzz
import PIL
import os
from PIL import ImageDraw
import asyncio
import concurrent.futures
import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import uniform_filter
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from skimage import img_as_ubyte
from skimage import restoration
import easyocr

from fastapi import FastAPI, File, status
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from io import BytesIO

from app import get_image_from_bytes
from app import detect_sample_model
from app import add_bboxs_on_img
from app import add_text_bboxs_on_img
from app import get_bytes_from_image
from app import return_bboxs_on_img
####################################### logger #################################

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

###################### FastAPI Setup #############################

# title

app = FastAPI(
    title="Object Detection FastAPI Template",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="2023.1.31",
)

# This function is needed if you want to allow client requests 
# from specific domains (specified in the origins argument) 
# to access resources from the FastAPI server, 
# and the client and server are hosted on different domains.
origins = [
    "http://localhost",
    "http://localhost:8008",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def save_openapi_json():
    '''This function is used to save the OpenAPI documentation 
    data of the FastAPI application to a JSON file. 
    The purpose of saving the OpenAPI documentation data is to have 
    a permanent and offline record of the API specification, 
    which can be used for documentation purposes or 
    to generate client libraries. It is not necessarily needed, 
    but can be helpful in certain scenarios.'''
    openapi_data = app.openapi()
    # Change "openapi.json" to desired filename
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)

# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get('/healthcheck', status_code=status.HTTP_200_OK)
def perform_healthcheck():
    '''
    It basically sends a GET request to the route & hopes to get a "200"
    response code. Failing to return a 200 response code just enables
    the GitHub Actions to rollback to the last version the project was
    found in a "working condition". It acts as a last line of defense in
    case something goes south.
    Additionally, it also returns a JSON response in the form of:
    {
        'healtcheck': 'Everything OK!'
    }
    '''
    return {'healthcheck': 'Everything OK!'}


######################### Support Func #################################

def crop_image_by_predict(image: Image, predict: pd.DataFrame(), crop_class_name: str,) -> Image:
    """Crop an image based on the detection of a certain object in the image.
    
    Args:
        image: Image to be cropped.
        predict (pd.DataFrame): Dataframe containing the prediction results of object detection model.
        crop_class_name (str, optional): The name of the object class to crop the image by. if not provided, function returns the first object found in the image.
    
    Returns:
        Image: Cropped image or None
    """
    crop_predicts = predict[(predict['name'] == crop_class_name)]

    if crop_predicts.empty:
        raise HTTPException(status_code=400, detail=f"{crop_class_name} not found in photo")

    # if there are several detections, choose the one with more confidence
    if len(crop_predicts) > 1:
        crop_predicts = crop_predicts.sort_values(by=['confidence'], ascending=False)

    crop_bbox = crop_predicts[['xmin', 'ymin', 'xmax','ymax']].iloc[0].values
    # crop
    img_crop = image.crop(crop_bbox)
    return(img_crop)


######################### MAIN Func #################################

@app.post("/licence_plate_location_number")
async def licence_plate_location_number(file: bytes = File(...)):
    timeout_seconds = 5

    # Define the timeout response
    def timeout_handler():
        return JSONResponse({"message": "Request timed out"}, status_code=408)

    # Define the actual processing logic
    def process_image():
        # get image from bytes
        input_image = get_image_from_bytes(file)

        # model predict
        predict = detect_sample_model(input_image)

        # add bbox on image
        try:
            final_image = return_bboxs_on_img(image=input_image, predict=predict)
        except Exception as e:
            print(e)
            return {"can't detect object"}

        # return image in bytes format
        return {"number": final_image[0],"province": final_image[1],"top": final_image[2],"left": final_image[3],"right": final_image[4],"bottom": final_image[5]}


    # Execute the processing logic with a timeout
    with concurrent.futures.ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        processing_task = loop.run_in_executor(executor, process_image)
        try:
            response = await asyncio.wait_for(processing_task, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            processing_task.cancel()
            response = timeout_handler()

    return response

@app.post("/img_object_detection_to_licence_number")
async def img_object_detection_to_img(file: bytes = File(...)):
    timeout_seconds = 5

    # Define the timeout response
    def timeout_handler():
        return JSONResponse({"message": "Request timed out"}, status_code=408)

    # Define the actual processing logic
    def process_image():
        # get image from bytes
        input_image = get_image_from_bytes(file)

        # model predict
        predict = detect_sample_model(input_image)

        # add bbox on image
        try:
            final_image = add_bboxs_on_img(image=input_image, predict=predict)
        except Exception as e:
            print(e)
            return {"can't detect object"}

        # return image in bytes format
        return StreamingResponse(content=get_bytes_from_image(final_image), media_type="image/jpeg")

    # Execute the processing logic with a timeout
    with concurrent.futures.ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        processing_task = loop.run_in_executor(executor, process_image)
        try:
            response = await asyncio.wait_for(processing_task, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            processing_task.cancel()
            response = timeout_handler()

    return response


