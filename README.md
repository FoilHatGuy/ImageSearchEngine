# ImageSearchEngine
Image Search engine for art gallery based on CV and NN

## File structure
main.py - flask framework methods\
ServerDetectorInterface.py - flask methods controller

imageSearcher - identification module

## imageSearcher functions interface
**request** - dict() containing field
- file            inputted image route

returns **response** - dict() containing following fields
- best            image id in database
- bestBatch       a range of best images
- result_file     base64 encoded best image
- src_file        base64 encoded input image
- desc            image description

## Config file fields description

  
## Config file fields description  
| field name    | description                                          | default    |
| ------------- | ---------------------------------------------------- | ---------- |
| inputSizes    | sizes the input image is resized before processing   | [400, 400] |
| PPResultSizes | sizes of image the preprocessing module outputs      | [240, 240] |
| workingFolder | folder with index.csv and pictures named \*id\*.jpeg | ---------- |
| localDBFolder | folder for detectors to write saved data             | ---------- |
| tempFolder    | folder for uploading input images                    | ---------- |
| host          | ip/hostname for flask to host seerver                | localhost  |
| numOfBest     | number of best images to show to user                | 5          |