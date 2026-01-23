# in main function
delay = 10 sec
not_full = True
new_entry = False

while not_full: # eventually we want to run this until the end of the program?
    cap_img()
    # run object detection and classification here 
    new_entry = categorize_img()
        takes in predictions.csv and compares with existing/previous body_parts.csv
        then outputs to new body_parts.csv
        every 10 minutes, the body_parts.csv file is uploaded to the cloud
    not_full = NF() # this is 90% accuracy and not full
    if !new_entry:
        delay += 10
