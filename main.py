import requests
import json
import numpy as np
import pandas as pd
import cv2 
import time

# Using the Python Device SDK for IoT Hub:
#   https://github.com/Azure/azure-iot-sdk-python
# The sample connects to a device-specific MQTT endpoint on your IoT Hub.
from azure.iot.device import IoTHubDeviceClient, Message, MethodResponse

CONNECTION_STRING = "{Your IoT hub device connection string}"
PREDICTION_URL="{Your Custom Vision Prediction URL}"
PREDICTION_KEY = "{Your Custom Vision Prediction Key}"
TAG_LIST = ["banana", "apple", "orange"]

def iothub_client_init():
    # Create an IoT Hub client
    client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
    return client

def cv_oj_api(url, key, img_file_path, img, tag_list):
    headers = {
            'content-type':'application/octet-stream',
            'Prediction-Key': key
            }
    response = requests.post(url, data = open(img_file_path, "rb"), headers = headers)
    response.raise_for_status()
    result = response.json()

    tag_grab_dict = {}
    #print(result)
    prob = pd.DataFrame([[tag_i["tagName"], tag_i["probability"]] for tag_i in result["predictions"]], columns=["tagName", "probability"])
    #print(prob.head)
    for tag in tag_list:
        if len(prob.query('tagName==@tag')) == 0:
            tag_grab_dict[(tag+'_px')] = np.nan
            tag_grab_dict[(tag+'_py')] = np.nan
            tag_grab_dict[(tag+'_x')] = np.nan
            tag_grab_dict[(tag+'_y')] = np.nan
            continue
        if prob.query('tagName==@tag')['probability'].max() < 0.5:
            tag_grab_dict[(tag+'_px')] = np.nan
            tag_grab_dict[(tag+'_py')] = np.nan
            tag_grab_dict[(tag+'_x')] = np.nan
            tag_grab_dict[(tag+'_y')] = np.nan
            continue

        tag_ids = prob.query('tagName==@tag')['probability']
        for i, probability in tag_ids.iteritems():
            if(probability > 0.5):
                tag_grab_dict['datetime'] = result['created']                
                tag_grid = result['predictions'][i]['boundingBox']
                y = int(tag_grid['top'] * img.shape[0])
                x = int(tag_grid['left'] * img.shape[1])
                h = int(tag_grid['height'] * img.shape[0])
                w = int(tag_grid['width'] * img.shape[1])
                # center of a box
                tag_x = x + int(w/2)
                tag_y = y + int(h/2)
                tag_grab_dict[(tag + '_px')] = x
                tag_grab_dict[(tag + '_py')] = y
                tag_grab_dict[(tag + '_x')] = tag_x
                tag_grab_dict[(tag + '_y')] = tag_y                

                cv2.putText(img, tag, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img, tag_grab_dict


if __name__ == '__main__':
    inpath = "./input/"
    outpath = "./output/"
    cap = cv2.VideoCapture(inpath + 'bulldozer_Trim.mp4') # Video file name to be predicted
    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter((outpath + 'result.mp4'), fourcc, 15, (cap_width, cap_height)) # Output file name

    count=0
    tag_grab_df = pd.DataFrame([])

    client = iothub_client_init()

    while(cap.isOpened()):
        t1 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        # 1 frame/sec 
        if count % fps != 0:
            continue

        # img resize
        #height = frame.shape[0]
        #width = frame.shape[1]
        #frame = cv2.resize(frame, (int(width*0.5), int(height*0.5)))

        tmp_file_path = outpath + 'tmp.jpg'
        cv2.imwrite(tmp_file_path, frame)

        # api prediction & masking
        frame, tag_grab_dict = cv_oj_api(PREDICTION_URL, PREDICTION_KEY, tmp_file_path, frame, TAG_LIST)

        # Send message to IoT Hub
        message = Message(json.dumps(tag_grab_dict))
        print( "Sending message: {}".format(message) )
        client.send_message(message)

        tag_grab_dict['time'] = count/fps
        tag_grab_df = tag_grab_df.append([tag_grab_dict])

        # write & show frame
        #print(frame.shape)
        out.write(frame)
        img_file_path = outpath + str(count) + '.jpg'
        #cv2.imwrite(img_file_path, frame)
        #cv2_imshow(frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        sec = count/fps
        if sec % 10 == 0:
            print(sec,'sec ended.')

        #for short time debag
        #if sec >= 10:
        #    break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    tag_grab_df.to_csv(outpath + 'tag_grab_df.csv', index=False, encoding='shift-JIS')