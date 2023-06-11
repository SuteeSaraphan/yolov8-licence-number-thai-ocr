from PIL import Image, ImageDraw, ImageFont
import io
import pandas as pd
import numpy as np
import easyocr
from fuzzywuzzy import fuzz

from typing import Optional

from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, colors


reader = easyocr.Reader(['th'])
# Initialize the models
model_sample_model = YOLO("./models/sample_model/best.pt")
font_path = "THSarabunNew.ttf"

# Load the Thai font
font2 = ImageFont.truetype(font_path, size=16)


def get_image_from_bytes(binary_image: bytes) -> Image:
    """Convert image from bytes to PIL RGB format
    
    Args:
        binary_image (bytes): The binary representation of the image
    
    Returns:
        PIL.Image: The image in PIL RGB format
    """
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image


def get_bytes_from_image(image: Image) -> bytes:
    """
    Convert PIL image to Bytes
    
    Args:
    image (Image): A PIL image instance
    
    Returns:
    bytes : BytesIO object that contains the image in JPEG format with quality 85
    """
    return_image = io.BytesIO()
    image.save(return_image, format='JPEG', quality=85)  # save the image in JPEG format with quality 85
    return_image.seek(0)  # set the pointer to the beginning of the file
    return return_image

def transform_predict_to_df(results: list, labeles_dict: dict) -> pd.DataFrame:
    """
    Transform predict from yolov8 (torch.Tensor) to pandas DataFrame.

    Args:
        results (list): A list containing the predict output from yolov8 in the form of a torch.Tensor.
        labeles_dict (dict): A dictionary containing the labels names, where the keys are the class ids and the values are the label names.
        
    Returns:
        predict_bbox (pd.DataFrame): A DataFrame containing the bounding box coordinates, confidence scores and class labels.
    """
    # Transform the Tensor to numpy array
    predict_bbox = pd.DataFrame(results[0].to("cpu").numpy().boxes.xyxy, columns=['xmin', 'ymin', 'xmax','ymax'])
    # Add the confidence of the prediction to the DataFrame
    predict_bbox['confidence'] = results[0].to("cpu").numpy().boxes.conf
    # Add the class of the prediction to the DataFrame
    predict_bbox['class'] = (results[0].to("cpu").numpy().boxes.cls).astype(int)
    # Replace the class number with the class name from the labeles_dict
    predict_bbox['name'] = predict_bbox["class"].replace(labeles_dict)
    return predict_bbox

def get_model_predict(model: YOLO, input_image: Image, save: bool = False, image_size: int = 1248, conf: float = 0.5, augment: bool = False) -> pd.DataFrame:
    """
    Get the predictions of a model on an input image.
    
    Args:
        model (YOLO): The trained YOLO model.
        input_image (Image): The image on which the model will make predictions.
        save (bool, optional): Whether to save the image with the predictions. Defaults to False.
        image_size (int, optional): The size of the image the model will receive. Defaults to 1248.
        conf (float, optional): The confidence threshold for the predictions. Defaults to 0.5.
        augment (bool, optional): Whether to apply data augmentation on the input image. Defaults to False.
    
    Returns:
        pd.DataFrame: A DataFrame containing the predictions.
    """
    # Make predictions
    predictions = model.predict(
                        imgsz=image_size, 
                        source=input_image, 
                        conf=conf,
                        save=save, 
                        augment=augment,
                        flipud= 0.0,
                        fliplr= 0.0,
                        mosaic = 0.0,
                        )
    
    # Transform predictions to pandas dataframe
    predictions = transform_predict_to_df(predictions, model.model.names)
    return predictions


################################# BBOX Func #####################################

def add_bboxs_on_img(image: Image, predict: pd.DataFrame()) -> Image:
    font_path = "THSarabunNew.ttf" # Replace "path_to_font.ttf" with the actual path to your font file
    font_size = 42
    font = ImageFont.truetype(font_path, size=font_size)
    draw = ImageDraw.Draw(image)
    predict = predict.sort_values(by=['xmin'], ascending=True)

    # get the coordinates of the first bounding box
    first_bbox = predict.iloc[0][['xmin', 'ymin', 'xmax', 'ymax']]
    left, top, right, bottom = first_bbox
    # crop the image
    cropped_image = image.crop((left, top, right, bottom))
    # Doing OCR. Get bounding boxes.
    np_image = np.array(cropped_image)
    bounds = reader.readtext(np_image,allowlist=" กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮเแะัํี๊้็่๋ิืุูึใไำ๑๒๓๔๕๖๗๘๙0123456789")
    country=""
    number = ''
    if(len(bounds) == 0):
        print("")
    else:
        for i, x in enumerate(bounds):
            if i < len(bounds) - 1:
                number += x[1].strip()
        place_names = ['กรุงเทพมหานคร', 'กระบี่', 'กาญจนบุรี', 'กาฬสินธุ์', 'กำแพงเพชร', 'ขอนแก่น', 'จันทบุรี', 'ฉะเชิงเทรา', 'ชลบุรี', 'ชัยนาท', 'ชัยภูมิ', 'ชุมพร', 'เชียงราย', 'เชียงใหม่', 'ตรัง', 'ตราด', 'ตาก', 'นครนายก', 'นครปฐม', 'นครพนม', 'นครราชสีมา', 'นครศรีธรรมราช', 'นครสวรรค์', 'นนทบุรี', 'นราธิวาส', 'น่าน', 'บึงกาฬ', 'บุรีรัมย์', 'ปทุมธานี', 'ประจวบคีรีขันธ์', 'ปราจีนบุรี', 'ปัตตานี', 'พระนครศรีอยุธยา', 'พังงา', 'พัทลุง', 'พิจิตร', 'พิษณุโลก', 'เพชรบุรี', 'เพชรบูรณ์', 'แพร่', 'ภูเก็ต', 'มหาสารคาม', 'มุกดาหาร', 'แม่ฮ่องสอน', 'ยโสธร', 'ยะลา', 'ร้อยเอ็ด', 'ระนอง', 'ระยอง', 'ราชบุรี', 'ลพบุรี', 'ลำปาง', 'ลำพูน', 'เลย', 'ศรีสะเกษ', 'สกลนคร', 'สงขลา', 'สตูล', 'สมุทรปราการ', 'สมุทรสงคราม', 'สมุทรสาคร', 'สระแก้ว', 'สระบุรี', 'สิงห์บุรี', 'สุโขทัย', 'สุพรรณบุรี', 'สุราษฎร์ธานี', 'สุรินทร์', 'หนองคาย', 'หนองบัวลำภู', 'อ่างทอง', 'อุดรธานี', 'อุทัยธานี', 'อุตรดิตถ์', 'อุบลราชธานี', 'อำนาจเจริญ'
]
        try:
            search_term = bounds[1][1].strip()
            best_match = max(place_names, key=lambda name: fuzz.ratio(name.lower(), search_term.lower()))
            country = best_match
        except Exception as e:
            print(e)
        print(number+" "+country)

    for _, row in predict.iterrows():
        # Create the text to be displayed on the image
        text = f"{'dasdกกกก'}: {int(row['confidence']*100)}%"
        # Get the bounding box coordinates
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        text_position = (bbox[0], bbox[1] - 40)
        # Add the bounding box on the image
        draw.rectangle(bbox, outline="red", width=2)
        # Add the text above the bounding box
        text_position2 = (bbox[0], bbox[1] - 45)
        bbvox = draw.textbbox(text_position2, number+" "+country, font=font)
        draw.rectangle(bbvox, fill="#76EEC6")
        draw.text(text_position, number+" "+country, fill="blue", font=font)

    # Perform additional operations with the draw object if needed

    return image

def return_bboxs_on_img(image: Image, predict: pd.DataFrame()) -> Image:
    font_path = "THSarabunNew.ttf" # Replace "path_to_font.ttf" with the actual path to your font file
    font_size = 42
    font = ImageFont.truetype(font_path, size=font_size)
    draw = ImageDraw.Draw(image)
    predict = predict.sort_values(by=['xmin'], ascending=True)

    # get the coordinates of the first bounding box
    first_bbox = predict.iloc[0][['xmin', 'ymin', 'xmax', 'ymax']]
    left, top, right, bottom = first_bbox
    print(left)
    print(top)
    print(right)
    print(bottom)

    # crop the image
    cropped_image = image.crop((left, top, right, bottom))
    # Doing OCR. Get bounding boxes.
    np_image = np.array(cropped_image)
    bounds = reader.readtext(np_image,allowlist=" กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮเแะัํี๊้็่๋ิืุูึใไำ๑๒๓๔๕๖๗๘๙0123456789")
    country=""
    number = ''
    if(len(bounds) == 0):
        print("")
    else:
        for i, x in enumerate(bounds):
            if i < len(bounds) - 1:
                print(x[1])
                number += x[1]

        place_names = ['กรุงเทพมหานคร', 'กระบี่', 'กาญจนบุรี', 'กาฬสินธุ์', 'กำแพงเพชร', 'ขอนแก่น', 'จันทบุรี', 'ฉะเชิงเทรา', 'ชลบุรี', 'ชัยนาท', 'ชัยภูมิ', 'ชุมพร', 'เชียงราย', 'เชียงใหม่', 'ตรัง', 'ตราด', 'ตาก', 'นครนายก', 'นครปฐม', 'นครพนม', 'นครราชสีมา', 'นครศรีธรรมราช', 'นครสวรรค์', 'นนทบุรี', 'นราธิวาส', 'น่าน', 'บึงกาฬ', 'บุรีรัมย์', 'ปทุมธานี', 'ประจวบคีรีขันธ์', 'ปราจีนบุรี', 'ปัตตานี', 'พระนครศรีอยุธยา', 'พังงา', 'พัทลุง', 'พิจิตร', 'พิษณุโลก', 'เพชรบุรี', 'เพชรบูรณ์', 'แพร่', 'ภูเก็ต', 'มหาสารคาม', 'มุกดาหาร', 'แม่ฮ่องสอน', 'ยโสธร', 'ยะลา', 'ร้อยเอ็ด', 'ระนอง', 'ระยอง', 'ราชบุรี', 'ลพบุรี', 'ลำปาง', 'ลำพูน', 'เลย', 'ศรีสะเกษ', 'สกลนคร', 'สงขลา', 'สตูล', 'สมุทรปราการ', 'สมุทรสงคราม', 'สมุทรสาคร', 'สระแก้ว', 'สระบุรี', 'สิงห์บุรี', 'สุโขทัย', 'สุพรรณบุรี', 'สุราษฎร์ธานี', 'สุรินทร์', 'หนองคาย', 'หนองบัวลำภู', 'อ่างทอง', 'อุดรธานี', 'อุทัยธานี', 'อุตรดิตถ์', 'อุบลราชธานี', 'อำนาจเจริญ'
]
        try:
            search_term = bounds[1][1].strip()
            best_match = max(place_names, key=lambda name: fuzz.ratio(name.lower(), search_term.lower()))
            country = best_match
        except Exception as e:
            print(e)

        print(number+"asd ")
        print(number+" "+country)
    res =  [number,country,str(top),str(left),str(right),str(bottom)]
    return res

def add_text_bboxs_on_img(image: Image, predict: pd.DataFrame(), textline) -> Image:
    font_path = "THSarabunNew.ttf" # Replace "path_to_font.ttf" with the actual path to your font file
    font_size = 36
    font = ImageFont.truetype(font_path, size=font_size)
    draw = ImageDraw.Draw(image)

    # Sort predict by xmin value
    predict = predict.sort_values(by=['xmin'], ascending=True)

    # Iterate over the rows of the predict dataframe
    for _, row in predict.iterrows():
        # Create the text to be displayed on the image
        text = f"{'dasdกกกก'}: {int(row['confidence']*100)}%"
        # Get the bounding box coordinates
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        text_position = (bbox[0], bbox[1] - 30)
        # Add the bounding box on the image
        draw.rectangle(bbox, outline="red", width=2)
        # Add the text above the bounding box
        draw.text(text_position, textline[0], fill="red", font=font)
        draw.text((text_position[0] + 1, text_position[1]), textline[0], fill="red", font=font)

    # Perform additional operations with the draw object if needed

    return image



################################# Models #####################################


def detect_sample_model(input_image: Image) -> pd.DataFrame:
    """
    Predict from sample_model.
    Base on YoloV8

    Args:
        input_image (Image): The input image.

    Returns:
        pd.DataFrame: DataFrame containing the object location.
    """
    predict = get_model_predict(
        model=model_sample_model,
        input_image=input_image,
        save=False,
        image_size=640,
        augment=False,
        conf=0.5,
    )
    return predict