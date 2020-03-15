# Dermabox - by Team Redbull-On-Rails
Detects atopic eczema in babies. Used by 3rd-world midwifery nurses.

## Inspiration
When a child is born in the developing world, its midwifery nurse lacks the proper training to detect whether the child has a skin disease. We wanted to create a low-cost, easy to build skin scanner, which is far cheaper than a doctor or current medical equipment. Such a device can alert whether the child has atopic eczema - a disease which is hard to detect. Our friends have told us of winning hackathon projects which targeted healthcare of developed nations, but we think it is a greater priority for humanity to improve the healthcare of developing nations.

## What it does
Dermabox is an skin scanner which uses OpenCV to scan and process the image of a child's skin. It uses machine learning to determine whether the child has atopic eczema - a skin disease which can be hard to detect and be easily missed by a busy nurse. Dermabox shows the results of the illness tests on an LCD. The results are sent to a database, where a doctor can take a look at them. Optionally, the results can be sent by Whatsapp message to a patient's parents.

## How we built it
We used Python to program our codebase. Our box is powered by a Raspberry Pi 3. We used TensorFlow, along with a Google Coral USB accelerator, to run our machine learning models at a suitable speed. The Dermabox contains a camera, a button and an LCD, which use the GPIO library to interact with the main program. The box itself was designed in Autodesk Fusion 360 and was 3D printed at nearby Ryerson University. PHP, MySQL and Twilio were used to store and deliver results if a child's parent wanted them to be sent to their phone. We designed a custom website using HTML, CSS and JavaScript, where a doctor or nurse can take a look at patient results.

## Challenges we ran into
It was very, very difficult to find datasets for our machine learning models. Many research studies had their datasets encrypted with password-protected zip files. To overcome this, we had to combine smaller datasets to train our model.

In addition, our raspberry pi had major firmware issues and odd SSL certificate problems. Luckily, after many hours, we were able to overcome them.

## Accomplishments that we're proud of
* For a hackathon project, our machine learning models actually work with a great accuracy: 75%!
* We made our own 3D model in autocad and then 3D printed it.
* We learned a lot about atopic eczema by reading through a couple of research papers.
* We finally got out Raspberry Pi to work with TensorFlow and Pandas.
* The idea itself incorporates really cool technology and has a meaningful purpose.
* ... and finally, the fact that no device on the market is able to take a scan of a person's skin and identify a skin disease! Usually, a doctor is required to perform this task, which takes a lot of time and a lot more money!

## What we learned
* How to prepare datasets.
* How to tweak hyperparameters for ML models.
* Skin biology.
* How to manage our time.
* How to integrate PHP, TensorFlow, OpenCV, GPIO, and other libraries into one, well structured program.
* How to design a nice-looking website.

## Categories we are competing for:
* Best hardware hack: Dermabox uses a Raspberry Pi 3, a camera, button and LCD. We used GPIO to communicate with our components. We also 3D modeled and 3D printed an enclosure for our hardware!
* Best web hack: Dermabox uses a web application designed for hospital/clinic staff to take a look at test results. The application uses PHP and MySQL as well as HTML, CSS and JavaScript.
* Best new hack: We thought outside the box by creating a low-cost device which does not share any similarities to what is on the market! Our device's goal is to make healthcare cheaper and better in developing nations, even though most health hackathon projects we have heard about from our peers target healthcare of developed nations.

## What's next for Dermabox
We want to add more ML models, which detect new diseases, such as seborrheic dermatitis. We also want to tweak our current ML models further, and train them on better (but more expensive) datasets. We were also thinking of creating a cardboard version of Dermabox, which (similar to Google Cardboard) can easily be built by anyone who owns the appropriate hardware components and the code.
