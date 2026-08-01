# EcoClasifica

Final project for the Building AI course

## Summary

EcoClasifica is a computer vision-based waste classification system that identifies whether an object is recyclable (plastic, glass, paper, or metal) or organic/non-recyclable from a photo, helping reduce contamination in recycling bins.

## Background

Incorrect waste sorting is one of the biggest challenges in recycling. When even a single non-recyclable item is placed in the wrong recycling bin, it can contaminate the entire batch, causing it to be sent to a landfill instead of being recycled. This happens frequently because many people are unsure which category certain items belong to (for example, a greasy pizza box or a coffee cup with a plastic lining).

* Contamination rates at recycling facilities exceed 25% in some countries.
* Lack of clear and accessible information at the moment people dispose of their waste.
* High costs associated with manual waste sorting in recycling plants.
* Loss of recyclable materials that end up being incinerated or sent to landfills.

My personal motivation came from realizing how often I hesitate when sorting my own recycling. If it happens to me, it probably happens to many other people as well. A simple assistant that only requires taking a photo could reduce that uncertainty and, at scale, significantly increase the amount of waste that is actually recycled.

## How is it used?

The user opens a mobile app (or uses a webcam at a community recycling station), takes a picture of the item they want to dispose of, and within seconds the system returns a label ("plastic," "glass," "paper/cardboard," "metal," or "organic/general waste") along with the appropriate recycling bin.

**Users and use cases:**
* People at home using their smartphones before throwing away waste.
* Community or university recycling stations equipped with a fixed camera above the recycling bins.
* Small waste sorting facilities that want an automated first inspection before manual sorting.

**Requirements to consider:**
* It should work under low-light conditions or with low-quality photos (not everyone has a good camera or proper lighting).
* The interface should be accessible to older adults and people with limited technological experience.
* It should provide a fast response (under 2–3 seconds) to avoid interrupting the recycling process.

A simplified example of the prediction code:

```python
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

CATEGORIES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'general_waste']

def classify_waste(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(np.expand_dims(x, axis=0))
    prediction = model.predict(x)
    category = CATEGORIES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return category, confidence

# category, confidence = classify_waste("waste_photo.jpg", model)
# print(f"This appears to be: {category} ({confidence:.1f}% confidence)")
```

## Data sources and AI methods

The project relies on labeled images of different types of waste. Some existing datasets that could be used individually or combined include:

* [TrashNet dataset](https://github.com/garythung/trashnet) — Approximately 2,500 labeled images across six waste categories.
* [Kaggle Garbage Classification dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) — Additional images to expand the training dataset.
* Custom photographs captured under different lighting conditions to improve the model's robustness.

| AI Technique | Purpose |
| ----------- | ----------- |
| Transfer learning using a pre-trained convolutional neural network (e.g., MobileNetV2) | Classify waste types from images while leveraging models trained on millions of images. |
| Data augmentation | Generate variations (rotation, brightness adjustment, cropping, etc.) so the model performs well on low-quality images. |
| Cross-validation | Evaluate how well the model generalizes to unseen images. |

MobileNetV2 is a good initial choice because it is lightweight and efficient enough to run on smartphones, making it suitable for offline use or areas with limited internet connectivity.

## Challenges

* The model cannot determine whether a recyclable item is actually clean (for example, a glass jar containing food residue), which also affects whether it can be recycled.
* It does not solve the underlying problem of excessive consumption and waste generation; it only helps improve waste separation.
* It depends on the availability of separate recycling bins, which are not available in many locations.
* The model may become biased if the training dataset does not include enough regional packaging variations (different brands, materials, or designs across countries).
* Ethical consideration: if deployed to automate large-scale waste sorting facilities, it could reduce jobs currently performed by human workers. Any such deployment should be designed to assist workers rather than replace them.

## Future work

* Expand the categories to include electronic waste and batteries, which require special handling.
* Integrate the classifier into a lightweight mobile application with offline functionality.
* Collaborate with local recycling centers or universities to test the system using real cameras and real-world data.
* Add geolocation features to help users find the nearest recycling bin or recycling center based on the detected material.
* To continue developing the project, I would need support in mobile app development, access to more locally labeled data, and possibly collaboration with a recycling facility to validate the system under real operating conditions.

## Acknowledgments

* [TrashNet dataset by Gary Thung and Mindy Yang](https://github.com/garythung/trashnet) — Primary dataset used for initial training.
* [Garbage Classification dataset on Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) — Additional training data.
* MobileNetV2 (Sandler et al., 2018) as the base architecture for transfer learning.
* Building AI project template created by Reaktor Innovations and the University of Helsinki.
