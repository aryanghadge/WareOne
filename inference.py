import torch
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# remove numbers that are too close to each other
def remove_close_numbers(numbers):
    numbers = np.sort(numbers)
    new_numbers = []
    for i, number in enumerate(numbers):
        if i == 0:
            new_numbers.append(number)
        else:
            if abs(number - new_numbers[-1]) > 40:
                new_numbers.append(number)
    return new_numbers


weights_1 = "CsPmB4E200.pt"
weights_2 = "cartonM_50e_16b.pt"
weights_3 = "cartons_palletM_best_200e_16b.pt"

bag_file = "test-flights-itc/flight-7.bag"

# custom model in weights folder
# model = torch.hub.load("../yolov5-7.0", "custom", path=weights, source="local")
# model.conf = 0.1  # confidence threshold (0-1)
# model.iou = 0.25  # NMS IoU threshold (0-1)

# model for detecting pallets
pallet_model = torch.hub.load("../yolov5-7.0", "custom", path=weights_1, source="local")
pallet_model.conf = 0.6  # confidence threshold (0-1)
pallet_model.iou = 0.25  # NMS IoU threshold (0-1)
# detect only class 1 (pallet)
pallet_model.classes = [1]

# model for detecting cartons
carton_model = torch.hub.load("../yolov5-7.0", "custom", path=weights_2, source="local")
carton_model.conf = 0.65  # confidence threshold (0-1)
carton_model.iou = 0.25  # NMS IoU threshold (0-1)
# detect only class 0 (carton)
carton_model.classes = [0]

# model.classes = None  # filter by class: --class 0, or --class 0 2 3

# realsense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Tell config that we will use 6 recorded device from file to be used by the pipeline through playback.
rs.config.enable_device_from_file(config, bag_file)

# Configure the pipeline to stream the depth stream
# Change this parameters according to the recorded bag file resolution
config.enable_stream(rs.stream.depth, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)

# record the session with timestamp in the name
# config.enable_record_to_file(f"test-{time.time()}.bag")

# Start streaming from file
pipeline.start(config)

# capture frames for video
output_frames = []
i = 0

try:
    while True:
        i += 1

        # Get frameset of depth
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Run inference
        pallet_results = pallet_model(color_image)
        carton_results = carton_model(color_image)

        # # create copies of the color image and the image to be rendered
        input_image = color_image.copy()
        output_image = color_image.copy()
        # # output_image = results.render()[0].copy()

        pallet_boxes = pallet_results.xyxy[0][pallet_results.pred[0][:, 5] == 1]
        print(pallet_boxes)

        filtered_pallet_boxes = []

        if len(pallet_boxes) > 0:
            # select the pallet which has the largest length
            filtered_pallet_boxes = pallet_boxes[
                np.argmax(pallet_boxes[:, 2] - pallet_boxes[:, 0])
            ]

        # filtered_pallet_boxes = pallet_boxes[
        #     np.argmax(pallet_boxes[:, 2] * pallet_boxes[:, 3])
        # ]

        pallet_boxes = [filtered_pallet_boxes]

        print(filtered_pallet_boxes)

        boxes_in_pallet = []
        layers_in_pallet = []

        if len(filtered_pallet_boxes) > 0:
            output_image = cv2.rectangle(
                output_image,
                (int(pallet_boxes[0][0]), int(pallet_boxes[0][1])),
                (int(pallet_boxes[0][2]), int(pallet_boxes[0][3])),
                (0, 255, 0),
                2,
            )

            # get the pallet coordinates
            x1_pallet = pallet_boxes[0][0] - 20
            y1_pallet = pallet_boxes[0][1]
            x2_pallet = pallet_boxes[0][2] + 20
            y2_pallet = pallet_boxes[0][3]

            # calculate depth of pallet
            depth_pallet = (
                np.mean(
                    depth_image[
                        int(pallet_boxes[0][1]) : int(pallet_boxes[0][3]),
                        int(pallet_boxes[0][0]) : int(pallet_boxes[0][2]),
                    ]
                )
                / 1000
            )

            # calculate the center of the pallet
            x_center = int((x1_pallet + x2_pallet) / 2)
            y_center = int((y1_pallet + y2_pallet) / 2)

            cv2.putText(
                output_image,
                "%.2f m" % depth_pallet,
                (x_center, y_center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            # count the number of boxes in the pallet
            boxes = carton_results.xyxy[0][carton_results.pred[0][:, 5] == 0]
            for box in boxes:
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]

                # check if the box is inside the pallet
                if x1 > x1_pallet and x2 < x2_pallet and y2 < y2_pallet:

                    # check if the box is in a new layer
                    if len(layers_in_pallet) == 0:
                        layers_in_pallet.append(y2)
                    else:
                        if abs(y2 - layers_in_pallet[-1]) > 40:
                            layers_in_pallet.append(y2)

                    boxes_in_pallet.append(box)
                    output_image = cv2.rectangle(
                        output_image,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 0, 255),
                        2,
                    )

                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)

                    # get depth from the center pixel
                    # depth = depth_image[y_center, x_center]

                    # get depth from the mean in the bounding box
                    depth = (
                        np.mean(
                            depth_image[
                                int(y1) : int(y2),
                                int(x1) : int(x2),
                            ]
                        )
                        / 1000
                    )

                    # depth = (
                    #     np.mean(
                    #         depth_image[
                    #             x_center - 10 : x_center + 10,
                    #             y_center - 10 : y_center + 10,
                    #         ]
                    #     )
                    #     / 1000
                    # )

                    # get depth from the smallest distance in the bounding box which is greater than 0
                    # ma = np.ma.masked_equal(
                    #     depth_image[
                    #         x_center - 10 : x_center + 10,
                    #         y_center - 10 : y_center + 10,
                    #     ],
                    #     0,
                    # )

                    # pth = np.min(ma) / 1000

                    # draw the depth on the image at the center of the bounding box
                    cv2.putText(
                        output_image,
                        "%.2f m" % depth,
                        (x_center, y_center),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

        # sort the layers from top to bottom
        layers_in_pallet = np.sort(layers_in_pallet)
        layers_in_pallet = remove_close_numbers(layers_in_pallet)

        # if there are layers that are too close to each other, remove the one that is closer to the top
        # if len(layers_in_pallet) > 1:

        print(layers_in_pallet)
        print(len(layers_in_pallet))

        # plot a horizontal line for each layer
        for layer in layers_in_pallet:
            output_image = cv2.line(
                output_image,
                (int(x1_pallet), int(layer)),
                (int(x2_pallet), int(layer)),
                (0, 255, 0),
                2,
            )

        cv2.imshow("Output", output_image)
        output_frames.append(output_image)
        # cv2.imshow("Output", carton_results.render()[0])

        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()

            # create and save a video from the frames
            height, width, layers = output_frames[0].shape
            size = (width, height)

            # get filename from the bag file
            filename = bag_file.split("/")[-1].split(".")[0]
            out = cv2.VideoWriter(
                f"{filename}_distance.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 15, size
            )

            for frame in output_frames:
                out.write(frame)
            out.release()
            break

finally:
    pipeline.stop()
