from flask import Flask, render_template, request,session, flash,redirect,url_for,send_file
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
import sys
sys.path.append('../')
import logomaker
import uuid
import shutil
import sys
import inspect

# name of the flask app
app = Flask(__name__)
# don't allow file transfers over 50 mb
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# this key decrypts the cookie on the client's browser
app.secret_key = os.urandom(32)

# allowed input file extensions
ALLOWED_EXTENSIONS = set(['txt','fasta','fa','csv','meme','input'])
ALLOWED_PARAM_EXTENSIONS = set(['txt'])

# handler methods for checking file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_param_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_PARAM_EXTENSIONS

'''
This is the default route that is hit when the default URL is entered into the browser. 
This handles data upload, logo draw, and parameter updates. Uploaded files are written to a 
single temporary file; the same will be true of the parameters. This route returns results in
an HTTP get when the URL is entered and results in a POST when parameters are updated. One every 
POST, the method re-renders the output page with the updated logo.  
'''
@app.route('/', methods=['GET','POST'])
def index():

    # dictionary to store all metadata
    # new keys/vals get written to file
    metaData = {}
    # radio button state and file format
    radioState = ''
    fileFormat = ''

    # real file name just for display.
    realUploadedFileName = ''

    # handle creation of temp files:
    # create 4 three temp files, input, style, metadata, log
    if 'uid' in session:
        print("Temp file already created")
        tempFile = str(session['uid']) + ".input"
        tempStyleFile = str(session['uid']) + ".txt"
        logFile = str(session['uid']) + ".log"
        metaDataFile = str(session['uid']) + ".meta"
        sys.stderr = WarningsLogger(logFile)
    else:
        session['uid'] = uuid.uuid4()
        tempFile = str(session['uid'])+".input"
        shutil.copy('crp_sites.fasta',tempFile)
        tempStyleFile = str(session['uid']) + ".txt"
        metaDataFile = str(session['uid']) + ".meta"
        logFile = str(session['uid']) + ".log"
        print("New temp file created")
        sys.stderr = WarningsLogger(logFile)

    # give the data file the temp name
    dataFileName = tempFile

    # name of the style file that gets passed onto make stylized logo
    # all parameter values are contained in this.
    style_file = tempStyleFile

    # default values of parameters
    default_parameters_text = """
character_colors : 'random'
logo_type : 'probability'
axes_type : 'classic'
font_name : 'Arial'
font_weight : 'heavy'
use_tightlayout : True"""

    # process the post from html
    if request.method == 'POST':

        # read radio state here via ajax post and write to metadata file
        # this is done to allow the user to change radio state after uploading
        fileFormatAjax = str(request.values.get('radioValue'))

        print(" In post ", fileFormatAjax, ' request info: ',request.form)

        metaData['session_id'] = str(session['uid'])

        if(fileFormatAjax !='None'):
            metaData['fileFormat'] = fileFormatAjax
            # write ajax metadata to file
            with open(metaDataFile, "w") as myfile:
                for key in sorted(metaData):
                    myfile.write(key + ":" + str("".join(metaData[key])) + "\n")

        # if draw logo button is hit, get updated values
        if str(request.form.get("parameterButton")) == 'Draw logo':

            updatedParametes = request.form['paramsTextArea']
            print("Hitting parameter update button ")
            with open(style_file, "w") as text_file:
                text_file.write(updatedParametes)

        # elif upload data button is hit, upload new data
        # but put new data in temp file
        elif str(request.form.get("dataUploadButton")) == 'Upload Data':
            print("Hitting Upload button ")

            # read radio button value
            if request.form.getlist('fileformat'):
                fileFormat = str(request.form['fileformat'])

                # record format in metadata dict, to be
                # written to file later
                metaData['fileFormat'] = fileFormat


            # get file name
            f = request.files['file']

            # display flash message if filetype not supported
            if not allowed_file(f.filename) and len(f.filename) is not 0:
                print(f.filename)
                flash(" File type not supported:  " + str(f.filename))
                # flash(" File type not supported ")

            # if button pressed without any uploaded
            elif len(f.filename) is 0:
                flash(" Please select a file to upload ")

            elif len(fileFormat) is 0:
                flash("Please choose a radio button")

            else:
               # secure filename cleans the name of the uploaded file
                # generate warning here if upload fails?
                f.save(secure_filename(f.filename))

                # write file name in metadata
                metaData['fileName'] = f.filename

                # write session name in metadata file for getting example from gallery
                metaData['session_id'] = str(session['uid'])

                # write all metadata to file
                with open(metaDataFile, "w") as myfile:
                    for key in sorted(metaData):
                        myfile.write(key + ":" + str("".join(metaData[key])) + "\n")

                # the following puts uploaded data in the temp file
                # write from
                with open(f.filename) as f1:
                    # write to
                    with open(dataFileName, "w") as f2:
                        for line in f1:
                            f2.write(line)

        # elif parameter upload button is pressed
        elif str(request.form.get("parameterUploadButton")) == 'Upload Parameters':
            print("Hitting parameter upload button ")

            # get file name
            f = request.files['file']

            # if button pressed without any uploaded
            if len(f.filename) is 0:
                flash(" Please select a parameters file to upload ")

            # if not right extension
            elif not allowed_param_file(f.filename) and len(f.filename) is not 0:
                print(f.filename)
                flash(" parameters must have .txt extension ")

            else:
                # secure filename cleans the name of the uploaded file
                f.save(secure_filename(f.filename))

                # the following puts uploaded data in the temp file
                # write from
                with open(f.filename) as f1:
                    # write to
                    with open(style_file, "w") as f2:
                        for line in f1:
                            f2.write(line)

        # check if example loaded from gallery. This is also a post
        # the following has lots of room for refactoring, or possibly relocating to another function
        elif 'example_number' in session:

            if session['example_number'] == 'Example 1':
                style_file = '/Users/tareen/Desktop/Desktop_Tests/logomaker_after_csv_parsing_params/logomaker/static/example_1_style.txt'
                dataFileName = '/Users/tareen/Desktop/Desktop_Tests/logomaker_after_csv_parsing_params/logomaker/static/example_1_data.fasta'
                metaData['fileFormat'] = 'fasta'

            elif session['example_number'] == 'Example 2':
                style_file = '/Users/tareen/Desktop/Desktop_Tests/logomaker_after_csv_parsing_params/logomaker/static/example_2_style.txt'
                dataFileName = '/Users/tareen/Desktop/Desktop_Tests/logomaker_after_csv_parsing_params/logomaker/static/example_2_data.fasta'
                metaData['fileFormat'] = 'fasta'

            elif session['example_number'] == 'Example 3':
                style_file = '/Users/tareen/Desktop/Desktop_Tests/logomaker_after_csv_parsing_params/logomaker/static/example_3_style.txt'
                dataFileName = '/Users/tareen/Desktop/Desktop_Tests/logomaker_after_csv_parsing_params/logomaker/static/example_3_data.fasta'
                metaData['fileFormat'] = 'fasta'

            elif session['example_number'] == 'Example 4':
                style_file = '/Users/tareen/Desktop/Desktop_Tests/logomaker_after_csv_parsing_params/logomaker/static/example_4_style.txt'
                dataFileName = '/Users/tareen/Desktop/Desktop_Tests/logomaker_after_csv_parsing_params/logomaker/static/example_4_data.txt'
                metaData['fileFormat'] = 'csv'

            # the following updates the temp data so gallery example keeps re-drawing.
            # write from
            with open(dataFileName) as f1:
                # write to
                with open(tempFile, "w") as f2:
                    for line in f1:
                        f2.write(line)

            # make necessary updates to metadata
            # write session name in metadata file for getting example from gallery
            metaData['session_id'] = str(session['uid'])
            # write all metadata to file
            with open(metaDataFile, "w") as myfile:
                for key in sorted(metaData):
                    myfile.write(key + ":" + str("".join(metaData[key])) + "\n")



    # show index page values on get
    elif request.method == 'GET':
        print("In get ")
        # on page load the first time, data file name is set to
        # the following file. This logo is displayed when the user
        # first arrives on the page

        # reset state/variables on get
        shutil.copy('crp_sites.fasta', tempFile)
        dataFileName = tempFile
        radioState = 'fasta'

        # reset metadata: describe why this is needed?
        if len(metaDataFile) != 0:
            open(metaDataFile, 'w').close()

        # write session name in metadata file for getting example from gallery
        metaData['session_id'] = str(session['uid'])

        # reset session if home-key pressed, so gallery example don't
        # intefere with creation of new logos
        if 'example_number' in session:
            session.pop('example_number', None)

        # write to metadata file
        with open(metaDataFile, "w") as myfile:
            for key in sorted(metaData):
                myfile.write(key + ":" + str("".join(metaData[key])) + "\n")

        # write the default parameter values to a temporary style file
        # which gets passed onto make_stylized_logo
        with open(style_file, 'w') as f:
            f.write(default_parameters_text)


    # read any meta data written to file
    # check if file exists first
    read_metadata_dict = {}
    try:
        with open(metaDataFile) as myfile:
            for line in myfile:
                name, var = line.partition(":")[::2]
                read_metadata_dict[name.strip()] = var
    except:
        print("No metadata file")

    if 'fileName' in read_metadata_dict:
        realUploadedFileName = str(read_metadata_dict['fileName']).strip()
        print("Real file name: ",realUploadedFileName)

    if 'fileFormat' in read_metadata_dict:
        radioState = str(read_metadata_dict['fileFormat']).strip()
        print("File format: ",radioState)

    # display parameter values in a textarea: 3 steps
    # Note: these steps could be combined into 1 step but
    # separated for clarity
    # 1) read raw text from style file

    with open(style_file, 'r') as p:
        rawParams = p.read()
    # remove any blank lines from parameters textarea
    rawParams = os.linesep.join([s for s in rawParams.splitlines() if s])

    # 2) store length of raw params in variable which
    # will be used in html
    paramsLength = len(rawParams)

    # 3) append tab delimited values to list so they look
    # exactly like the raw data in the parameters file
    displayParams = []
    for index in range(paramsLength):
        displayParams.append(rawParams[index].split('    '))


    # display input data in input text area
    # Also in 3 steps (see comments above)

    # 1) read raw text from fasta file
    with open(dataFileName, 'r') as fileVar:
        rawInput = fileVar.readlines()

    # 2) store length of raw data in variable which
    # will be used in html
    inputDataLength = len(rawInput)

    # 3) append tab delimited values to list so they look
    # exactly like the raw data in the parameters file
    displayInput = []
    for x in range(inputDataLength):
        displayInput.append(rawInput[x].split('    '))

    # this clears the plot before re-rendering so old logos
    # aren't drawn on top of each other.
    plt.cla()

    logomaker.make_styled_logo(style_file=style_file, fasta_file=dataFileName)
    logoFailure = ''

    '''
    # backend calls
    try:
        logoFailure = ''    # zero length for this variables means logo rendered w/o failure
        if radioState == 'fasta':
            print("Calling logomaker from fasta", dataFileName)
            logomaker.make_styled_logo(style_file=style_file,fasta_file=dataFileName)
        elif radioState == 'meme':
            print("Calling logomaker from meme",dataFileName)
            logomaker.make_styled_logo(style_file=style_file,meme_file=dataFileName)
        elif radioState == 'csv':
            #logomaker.make_styled_logo(style_file=style_file, csv_file=dataFileName)
            logomaker.make_styled_logo(style_file=style_file, matrix_csvfile=dataFileName)
        else:
            print("Calling logomaker from default", dataFileName, " radiostate: ",radioState)
            logomaker.make_styled_logo(style_file=style_file, fasta_file=dataFileName)
    except:
        # display this message in lieu of failed logo rendering.
        logoFailure = "ERROR: Could not draw OldLogo"
    '''

    # save the logo as a stream of bytes which can be passed into
    # the template. This is what's displayed on the website.
    logoFigFile = BytesIO()
    plt.savefig(logoFigFile, format='png')
    logoFigFile.seek(0)

    # the following contains the actual data passed to the html template
    logoFigData = base64.b64encode(logoFigFile.getvalue())

    # Show warnings
    with open(logFile) as log:
        flash(log.read())
    cleanWarnings(logFile)

    # form default values dict
    param_dict = logomaker.documentation_parser.parse_documentation_file('make_logo_arguments.txt')
    default_values = inspect.getargspec(logomaker.make_logo)
    doc_dict = dict(zip(default_values[0], list(default_values[3])))

    # for unique sections names
    sectionSet = set()
    # form dictionary with default values, descriptions, and sections
    doc_dict_2 = {}
    param_pairs = [(val.param_num, val.section, val.name, val.description) for val in param_dict.values()]
    for num, section, name, description in sorted(param_pairs):
        doc_dict_2[name] = (doc_dict[name], description, section)
        sectionSet.add(section)

    # change to list to access section as elements
    sectionList = sorted(list(sectionSet))
    sectionIndex = 0  # index to iterate unique sections
    # dictionary for showing parameters based on sections
    sectionDict = {}

    # sort by section; index 1 is value, 2 is section.
    for key, value in sorted(doc_dict_2.items(), key=lambda x: x[1][2]):
        # section matches unique seciton in set, return all parameters associated with it
        if (value[2] == sectionList[sectionIndex]) and sectionIndex < len(sectionList):
            # sectionList[sectionIndex] is going to be button name and id
            # key are going to be table elements
            # print(sectionList[sectionIndex], key, value[0], value[1])

            # new dict with section name as key and values being all the associated parameter names.
            # along with default values and descriptions
            if sectionList[sectionIndex] in sectionDict:
                sectionDict[sectionList[sectionIndex]].append([key, value[0], value[1]])
            else:
                sectionDict[sectionList[sectionIndex]] = [[key, value[0], value[1]]]
        else:
            sectionIndex += 1


    # for downloads
    localpath = '/Users/tareen/Desktop/Desktop_Tests/logomaker_after_csv_parsing_params/logomaker/static/'
    plt.savefig(localpath+str(session['uid'])+'.png')
    plt.savefig(localpath+str(session['uid'])+'.pdf')
    plt.savefig(localpath+str(session['uid'])+'.ps')
    plt.savefig(localpath+str(session['uid'])+'.svg')


    plt.close('All')
    # render the template with logo data
    return render_template('output.html', result=logoFigData, paramsLength=paramsLength, displayParams=displayParams,
                           displayInput=displayInput, inputDataLength=inputDataLength, doc_dict=doc_dict_2,sectionDict=sectionDict,
                           radioState=radioState, logoFailure=logoFailure, realUploadedFileName=realUploadedFileName,tempFileName=str(session['uid']))


# return to home page after click homepage button
@app.route('/home',methods=['GET','POST'])
def redirect_index():

    # reset session if home-key pressed, so gallery example don't
    # intefere with creation of new logos
    if 'example_number' in session:
        session.pop('example_number', None)
    return redirect(url_for('index'))

# download logo in the format chosen by user
@app.route('/download_logo/')
@app.route('/download_logo/<imageFormat>')
def download_logo(imageFormat='png'):
    print("xxx")
    print(imageFormat)
    print("xxx")

    return send_file('/Users/tareen/Desktop/Desktop_Tests/logomaker_after_csv_parsing_params/logomaker/static/'+str(session['uid'])+'.'+imageFormat)

# return to home page after click homepage button
@app.route('/gallery',methods=['GET','POST'])
def gallery():
    return render_template('gallery.html')

# return to home page after click homepage button
@app.route('/help',methods=['GET'])
def help():
    return render_template('help.html')

# return to home page after click homepage button
@app.route('/about',methods=['GET'])
def about():
    return render_template('about.html')

# get gallery example
@app.route('/getGalleryExample',methods=['GET','POST'])
def getGalleryExample():

    # process the post from html
    if request.method == 'POST':

        print(type(request.form))


        if 'example_1_button' in request.form:
                print('redirecting to index with example 1')

                # set gallery example in the session dict
                session['example_number'] = 'Example 1'
                return redirect(url_for('index',example_number=session['example_number']),code=307)

        elif 'example_2_button' in request.form:

            print('redirecting to index with example 2')
            # set gallery example in the session dict
            session['example_number'] = 'Example 2'
            return redirect(url_for('index',example_number=session['example_number']),code=307)


        elif 'example_3_button' in request.form:
            print('redirecting to index with example 3')

            # set gallery example in the session dict
            session['example_number'] = 'Example 3'
            return redirect(url_for('index',example_number=session['example_number']),code=307)


        elif 'example_4_button' in request.form:
            print('redirecting to index with example 4')

            # set gallery example in the session dict
            session['example_number'] = 'Example 4'
            return redirect(url_for('index',example_number=session['example_number']),code=307)

        return request.form

@app.before_first_request
def before_first_request():

    '''
    # attempt to create unique session id
    # and make temp file
    # if it's not made here, for some weird server issue, make again in index
    session['uid'] = uuid.uuid4()
    tempFile = str(session['uid']) + ".input"
    shutil.copy('crp_sites.fasta', tempFile)
    '''
    pass


# for broken or miss-typed links
@app.errorhandler(404)
def broken_link(e):
    return render_template('four_0_four.html'), 404

# for file sizes bigger than 50 mb
@app.errorhandler(413)
def file_too_large(e):
    return render_template('file_too_large.html'), 413


@app.before_request
def before_request():
    # here we can do something before every
    # request is made
    pass


def cleanWarnings(logFile):
    # erase warnings file before request
    if len(logFile)!=0 :
        open(logFile,'w').close()


class WarningsLogger(object):

    def __init__(self, logFileName):
        self.terminal = sys.stdout
        #self.log = open("warnings.log", "a")
        self.log = open(logFileName, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        # why flush, see following:
        # https://stackoverflow.com/questions/24011117/logging-realtime-stdout-to-a-file-in-python
        self.terminal.flush()
        self.log.flush()

if __name__ == "__main__":
    app.run(debug=True)
    #app.run()
