import pandas as pd

def MammogramSlidingWindows_reverse(imgdir, csvpath, savecsvdir):
    """
    imgdir = directory that contains the resized input images to "new_height * 640"
    csvpath = path of csv that contains the predicted bboxes on the input images
    savecsvdir = path to save the new csv that contains the inferenced bboxes on the resized input images"""
    imagelist = os.listdir(imgdir)
    df = pd.read_csv(csvpath)

    result = []

    for imagefile in imagelist:
        imagepath = imgdir + '/' + f'{imagefile[:-4]}.jpg'
        # print(imagepath)
        image_id = f'{imagefile[:-4]}.jpg'
        image = cv2.imread(imagepath,cv2.IMREAD_GRAYSCALE)
        image_height, image_width = image.shape




        filtered_df = df[df['image_id'].str.startswith(image_id[:-4])]

        num_splits = image_height // image_width + 1

        step = (image_height - image_width) // num_splits



        for index, row in filtered_df.iterrows():
            SlidingWindowName = row['image_id']
            SplitIndex = int(SlidingWindowName[:-4][-1])

            x_s = row['xmin']
            y_s = row['ymin']
            w_s = row['xmax'] - row['xmin']
            h_s = row['ymax'] - row['ymin']
            lesion_s = row['lesion_types']
            birads_s = row['birads']
            conf_lesion_s = row['conf_lesion']
            conf_birads_s = row['conf_birads']
            # print(lesion_s+"_"+birads_s)

            if x_s == '' and y_s == '' and w_s == '' and h_s == '':
                continue
            else:
                y_o = y_s + (step * SplitIndex)
                x_o = x_s
                w_o = w_s
                h_o = h_s
                lesion_o = lesion_s
                birads_o = birads_s
                conf_lesion_o = conf_lesion_s
                conf_birads_o = conf_birads_s
                conf_final_o = conf_lesion_o*0.5 + conf_birads_o*0.5
                result.append([image_id,x_o,y_o,w_o,h_o,lesion_o,birads_o,conf_final_o])
    resultDF = pd.DataFrame(result,columns = ['image_id','x','y','width','height','lesion_types','birads','confidence'])
    resultDF.to_csv(savecsvpath + '/' + "Inferenced_results.csv",index=False)

