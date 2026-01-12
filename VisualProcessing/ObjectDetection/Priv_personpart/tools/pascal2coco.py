import os
import json
import glob
import cv2
import numpy as np
import xml.etree.ElementTree as ET


_classes = ('__background__',  # always index 0
            'person', 'face', 'hand')

data_list = open('../ImageSets/privpersonpart_train.txt', 'r').read().splitlines()
im_path = '../Images/'
xml_path = '../Annotations/'
save_json_path = '../Json_Annos/privpersonpart_train.json'


class Pascal2coco(object):
    def __init__(self, _data_list, _save_json_path):
        self.data_list = data_list
        self.save_json_path = save_json_path

        self.images = []
        self.categories = []
        self.annotations = []

        self.label_map = {}
        for i in range(len(_classes)):
            self.label_map[_classes[i]] = i

        self.annID = 1
        
        self.transfer_process()
        self.data2coco()

    def transfer_process(self):
        # categories
        for i in range(1, len(_classes)):
            categories = {'supercategory': _classes[i], 'id': i,
                          'name': _classes[i]}

            self.categories.append(categories)

        print(self.categories)

        for num, data_name in enumerate(self.data_list):
            if num % 100 == 0 or num + 1 == len(self.data_list):
                print('XML transfer process  {}/{}'.format(num + 1, len(self.data_list)))

            # XML
            xml_file = glob.glob(xml_path + data_name + '.xml')
            tree = ET.parse(xml_file[0])
            filename = tree.find('filename').text + '.jpg'
            filename = filename.replace('.jpg.jpg', '.jpg')
            size = tree.find('size')

            # image
            im = cv2.imread('{}{}'.format(im_path, filename))
            height = im.shape[0]
            width = im.shape[1]

            # images
            image = {'height': height, 'width': width, 'id': num + 1, 'file_name': filename}
            self.images.append(image)

            object = tree.findall('object')
            for ix, obj in enumerate(object):
                bbox = obj.find('bndbox')

                label = obj.find('name').text.lower().strip()
                if label == 'none':
                    print(filename)
                    continue

                x1 = np.maximum(0.0, float(bbox.find('xmin').text))
                y1 = np.maximum(0.0, float(bbox.find('ymin').text))
                x2 = np.minimum(width - 1.0, float(bbox.find('xmax').text))
                y2 = np.minimum(height - 1.0, float(bbox.find('ymax').text))

                # rectangle = [x1, y1, x2, y2]
                bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]  # [x,y,w,h]
                area = (x2 - x1 + 1) * (y2 - y1 + 1)

                # annotations
                annotation = {'segmentation': [], 'iscrowd': 0, 'area': area, 'image_id': num + 1,
                              'bbox': bbox,
                              'category_id': self.label_map[label], 'id': self.annID}
                self.annotations.append(annotation)
                self.annID += 1

    def data2coco(self):
        data_coco = {'images': self.images, 'categories': self.categories, 'annotations': self.annotations}

        json.dump(data_coco, open(self.save_json_path, 'w'), indent=4)
        

if __name__ == '__main__':
    Pascal2coco(data_list, save_json_path)

