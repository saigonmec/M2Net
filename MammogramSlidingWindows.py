import os
import pandas as pd
import cv2

def crop_small_images_containing_bbox(csv_file_path,save_img_dir,img_dir):
    # Load the CSV file containing bounding box information
    df = pd.read_csv(csv_file_path)
    # Create new folder to write the images
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    # Create a list to save information
    new_info = []
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Load the original image using OpenCV
        original_image_id = row['image_id']
        original_image_path = os.path.join(img_dir,original_image_id)
        original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

        # Original image dimensions
        image_height = original_image.shape[0]
        image_width = original_image.shape[1]

        # Bounding box coordinates
        bbox_x, bbox_y, bbox_w, bbox_h = row['x'], row['y'], row['width'], row['height']
        bbox_label1 = row['lesion_types']
        bbox_label2 = row['birads']

        # bbox_label = "Abnormality" #comment chỗ này nếu chỉ muốn ra 1 label Abnormality

        # Rest of your code to split the image and process the bounding boxes
        # ... (same as before)
        # Split parameters
        split_height = image_width
        split_width = image_width
        # Calculate the number of splits
        num_splits = image_height // split_height + 1

        start = 0
        stop = image_height - image_width + 1
        step = (image_height - image_width) // num_splits

        # print('step: ', step)
        # Iterate over the splits
        for y in range(start, stop, step):
            # Calculate the starting and ending positions for each split
            # y is y
            end_y = y + split_height

            # Extract the split as a small image
            small_image = original_image[y:end_y, 0:image_width]

            

            # Calculate the intersection of the bounding box and the small image
            intersection_x = max(bbox_x, 0)
            intersection_y = max(bbox_y - y, 0)
            intersection_w = min(bbox_x + bbox_w, image_width) - intersection_x
            intersection_h = min(bbox_y + bbox_h, end_y) - y - intersection_y

            if intersection_w > 0 and intersection_h > 0:
                # Save the small_image that contains the bbox
                cv2.imwrite(save_img_dir + '/' + f"{original_image_id[:-4]}_{y//step}.jpg", small_image)
                # Compute the new bounding box coordinates relative to the small image
                new_bbox_x = int(intersection_x)
                new_bbox_y = int(intersection_y)
                new_bbox_w = int(intersection_w)
                new_bbox_h = int(intersection_h)

                # Append info
                new_info.append(
                    [f"{original_image_id[:-4]}_{y//step}.jpg",
                    split_width,
                    split_height,
                    new_bbox_x,
                    new_bbox_y,
                    new_bbox_w,
                    new_bbox_h,
                    bbox_label1,
                    bbox_label2])

            # if intersection_w <= 0 or intersection_h <= 0:
            #     # Compute the new bounding box coordinates relative to the small image
            #     new_bbox_x = int(intersection_x)
            #     new_bbox_y = int(intersection_y)
            #     new_bbox_w = int(intersection_w)
            #     new_bbox_h = int(intersection_h)


            #     # Append info
            #     new_info.append(
            #         [f"{original_image_id[:-4]}_{y//step}.jpg",
            #         split_width,
            #         split_height,
            #         '',
            #         '',
            #         '',
            #         '',
            #         '',
            #         ''])

    new_df = pd.DataFrame(new_info,columns=['image_id','image_width','image_height','x','y','width','height','lesion_types','birads'])
    new_df.to_csv('/content' + '/' + 'InferenceImages.csv',index=False)
