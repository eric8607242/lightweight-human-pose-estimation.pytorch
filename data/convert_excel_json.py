import pandas as pd
import json , os, argparse

def parse_data(initial_json,excel,image_path,save_path,image_height,image_width):
    with open(initial_json) as f:
        json_file = json.load(f)

    # initial 
    data = pd.read_excel(excel) 
    df = pd.DataFrame(data)

    #image
    license = 1
    file_name = ""
    height = image_height
    width = image_width
    image_id = 0

    #annotation
    segmentation = [[0]]
    iscrowd = 0
    num_key = 17
    area = 0
    keypoint= []
    bounding_box = []
    category_id = 1
    id = 1


    for img in os.listdir(image_path):
        temp = df[df["imgname"]==img]
        image_id = int(os.path.splitext(img)[0])

        #images
        json_file["images"].append({'license':int(license),
                                    'file_name':img,
                                    'height':int(height),
                                    'width':int(width),
                                    'id':int(image_id)})

        #keypoint
        for i in range(1,18):
            keypoint.append(int(temp["kp"+str(i)+"_x"].values[0]))
            keypoint.append(int(temp["kp"+str(i)+"_y"].values[0]))
            keypoint.append(int(2))

        #bounding box
        bounding_box.append(int(temp["bndbox_x0"].values[0]))
        bounding_box.append(int(temp["bndbox_y0"].values[0]))
        bounding_box.append(int(temp["bndbox_x1"].values[0]))
        bounding_box.append(int(temp["bndbox_y1"].values[0]))

        area = (int(temp["bndbox_x1"].values[0])-int(temp["bndbox_x0"].values[0])) * (int(temp["bndbox_y1"].values[0])-int(temp["bndbox_y0"].values[0]))
        print(area)

        json_file["annotations"].append({'segmentation':segmentation,
                                         'iscrowd':int(iscrowd),
                                         'num_keypoints':int(num_key),
                                         'area':int(area),
                                         'image_id':int(image_id),
                                         'keypoints':keypoint,
                                         'bbox':bounding_box,
                                         'category_id':int(category_id),
                                         'id':int(image_id+1000)})
        bounding_box=[]
        keypoint=[]

    json_file = json.dumps(json_file)
    
    with open(save_path, 'w') as f:
        for line in json_file:
            f.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_JSON_path', type=str, required=True,help='path to the initial JSON file')
    parser.add_argument('--excel_file_path', type=str, required=True, help='path to excel file')
    parser.add_argument('--image_path', type=str, required=True, help='path to image folder')
    parser.add_argument('--save_file_path', type=str, required=True, help='path to save JSON file')
    parser.add_argument('--image_height', type=int, default=256, help='image height')
    parser.add_argument('--image_width', type=int, default=320, help='image width')
    args = parser.parse_args()

    parse_data(args.initial_JSON_path,args.excel_file_path,args.image_path,args.save_file_path,args.image_height,args.image_width)
