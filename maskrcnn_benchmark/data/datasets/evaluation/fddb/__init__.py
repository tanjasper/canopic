from data_fns import get_paths
import os


def fddb_evaluation(predictions=None, dataset=None, output_folder=None, pixel_offset=0, **kwargs):

    print('FDDB evaluation -- applying pixel offset of {}'.format(pixel_offset))

    fnames = get_paths(dataset.filenames_path)  # get relative path fddb filenames (not absolute path)
    fnames = [x.replace('.jpg', '') for x in fnames]

    output_file = os.path.join(output_folder, 'fddb_predictions.txt')

    with open(os.path.join(output_folder, 'fddb_readme.txt'), 'w') as fp:
        fp.write("Pixel offset: {}".format(pixel_offset))

    with open(output_file, 'w') as fp:
        for image_id, prediction in enumerate(predictions):

            # format predictions to xywh
            img_info = dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            prediction = prediction.resize((image_width, image_height))
            prediction = prediction.convert("xywh")

            # convert predictions to list
            # pixel_offset is padding applied to the image by maxpool
            boxes = prediction.bbox.tolist()
            boxes = [[b[0] - pixel_offset, b[1] - pixel_offset, b[2], b[3]] for b in boxes]
            scores = prediction.get_field("scores").tolist()

            # write fddb predictions
            fp.write("{}\n".format(fnames[image_id]))  # filenames
            fp.write("{}\n".format(len(boxes)))  # number of predictions
            for i in range(len(boxes)):  # x y w h score for each prediction
                fp.write("{} {} {} {} {}\n".format(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i]))
