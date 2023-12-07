# create layer class with the following attributes:
# - layer number
# - boxes in layer
# - layer height

# the layer is initialized with zero boxes


class Layer:
    def __init__(self, layer_number, layer_height):
        self.layer_number = layer_number
        self.boxes_in_layer = []
        self.layer_height = layer_height

    def __str__(self):
        return f"Layer {self.layer_number} with {len(self.boxes_in_layer)} boxes"

    def __repr__(self):
        return f"Layer {self.layer_number} with {len(self.boxes_in_layer)} boxes"

    def get_layer_number(self):
        return self.layer_number

    def get_boxes_in_layer(self):
        return self.boxes_in_layer

    def get_layer_height(self):
        return self.layer_height

    def set_layer_number(self, layer_number):
        self.layer_number = layer_number

    def set_boxes_in_layer(self, boxes_in_layer):
        self.boxes_in_layer = boxes_in_layer

    def set_layer_height(self, layer_height):
        self.layer_height = layer_height

    def add_box_to_layer(self, box):
        self.boxes_in_layer.append(box)

    def remove_box_from_layer(self, box):
        self.boxes_in_layer.remove(box)

    def get_number_of_boxes_in_layer(self):
        return len(self.boxes_in_layer)

    def get_layer_height(self):
        return self.layer_height
