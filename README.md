# WeedNet
A Modern Farming Platform With an Integrated State of the Art Weed Detector.

## Weed Detector

WeedNet is a two-part project. The first is a state of the art deep learning powered image classifier that can classify close-up images of crops (as taken from a tractor or other farm equipment) into either 8 different weed categories or the not-weed category.

The dataset used, deepweeds, is from [a paper published in nature](https://www.nature.com/articles/s41598-018-38343-3). The original authors are able to acheive a peak accracy of ~95.7%. Our best model achieves ~98.03% on the same dataset.

We thank the AWS team and Agco for providing accress to remote instances that allowed us to launch distributed training jobs on multiple GPUs, that were crucial for our rapid iteration during the Agco 2020 hackathon.

[TABLE]

## Famer Interface

Along with the weed detector, we have developed an intuitive user interface for farmers to use our model as well as other interesting visualizations based on barn data that the farmers have. The web app allows farmers to remotely monitor crucial information from their barns and fields in real time, and can easily be extended on a case-by-case basis to allow farmers to perform actions based on that data. Some possible uses of this are remote deployment of fertilizers and pesticides, dynamic control of barn conditions, and mobile notifications during times of emergency.

The barn data visualization tools were tailored towards specific, real-world barn data. In order to preserve the privacy of the farmers, we cannot release this dataset. However, all code is open source, and our tools can be easily modified to work with custom barn data in various formats.
