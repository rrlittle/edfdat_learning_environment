'''
    This module is all about parsing, processing and creating dat files as defined 
    here : http://www.neurophys.wisc.edu/comp/docs/daflib/

    __ dat file description __
    A dat file is a collection of 512 byte blocks. 
    the first set of blocks is called a directory.

    The directory catalogs the datasets residing within the
    datafile. 

    The directory is composed of a 64 byte header and any number
    of dataset headers which are each 32 bytes. It can also 
    cross any number of blocks, but the size of the directory is
    always an integer number of blocks and the dataset headers are contiguous.

    After the directory come the datasets, which are referred to by the 
    dataset headers in the directory. each dataset has a mandatory header
    at it's start as well. but depending on the schema that dataset uses 
    it can be different. 


    __ USAGE __
    At the moment you should just run this script either in idle
    or in an interactive terminal using the command
    `python -i parsedat.py`

    Running it will automatically parse swigert_391.dat if it's 
    in the same directory. Why swigert_391 you ask?
    well it's just out of convenience. All the dat files should be 
    syntacticly correct and I'm still trying to understand the syntax.

    At the moment the script will print out all the variables parsed 
    from the dat file
    and leave you pointing to a tricky location, the TSDATA integer vector.

    You have a coule useful things availble to you.
    enter `locals().keys()` to print all the items this makes 
    available to you. 
    f 		---	a file pointer to the open dat file
    safe  	---	a integer, the first unkown byte of the file. i.e. 
    		use f.seek(safe) to return the file pointer to the
    		start of the TSDATA integer vector
    bin2int(binary string)	---	use bin2int(f.read(4)) to turn the binary
    			into a useful, recognizable integer
    			There are a number of these. like bin2int,bin2str,bin2short
	AND MORE! 


    __ NOTES __ 
    It might be useful to note that blocks are indexed from 1. 
    So to seek to the 51'st block use. f.seek(block_size * 51 - 1)
    Because python indexes starting at 0. 


     __ The Global data in this module are schema for headers and things __ 
     These are the header schemes.
     they are an ordered list of tuples in the form (description, size (in bytes), parsing function)
      The description is a string vaguely describing what the field is
      The size refers to the number of bytes the field takes
      The parsing function is te name of a function that will parse the field. such that 
          ```parsing_function( file_pointer.read(size) )``` will return the value in a
          a reasonable python type. like str, int, datetime.datetime, etc.  

    This supports these standard forms for schema definitions
    1.
        [(field name, size, parsing function),
        ...]
    2.
        [(field name, comment, size, parsing function),
        ...]

    schemas _MAY_ mix and match these standard forms.


'''
import struct as s
import binascii as ba

# CONVERIONS
def blocks2bytes(blocks): # converts from 512 byte blocks to bytes
    ''' converts from 512 byte blocks to bytes'''
    return blocks * block_size_bytes
def words2bytes(words): # converts from 32 bit words to bytes
    ''' converts from 32 bit words to bytes'''
    return words * 4 # 32 bits --> 4 bytes
def bin2hex(binstr,size=None):
    ''' converts binary string to hex value seperated by colon'''
    hx = ba.hexlify(binstr)
    return hx
def bin2pointer(binstr, size=None): # converts binary string to C pointer
    '''converts binary string to C pointer'''
    return s.unpack('P',binstr)[0]
def bin2float(binstr, size=None): # converts a binary to a float
    ''' converts a binary string to a float'''
    return s.unpack('f',binstr)[0]
def bin2int(binstr, size=None): # converts a binary string to an int
    ''' converts a binary string to integer'''
    if size:
    	try:
    		return s.unpack('i' + str(size),binstr)[0] # normally returns a tuple. we want an int
    	except:
    		if size is 16:
    			return s.unpack('l',binstr)[0] # try unpacking to a long
    else:
    	return s.unpack('i',binstr)[0] # normally returns a tuple. we want an int
def bin2str(binstr, size=None): # converts a binary string to a str
    ''' converts binary string to python string '''
    str(binstr) # python defaults to converting bin data to ascii   
def bin2short(binstr, size=None,read=False): # converts a binary string to a short
    '''converts a binary string to a short'''
    if read: print binstr
    return s.unpack('h',binstr)[0]
def com_vartype_convert(binstr, typecode, size=None, verb=True): # converts a binary string to a type based on the com type
    '''
        COM DAT files contain varibales that can be of a number of different types.
        they use codes to identify what the type is. 

        1 = int,
        2 = fp,
        3 = char

        I have not run into a fp yet. So I don't know how to handle it. 
        they throw an exception 
    '''
    #print 'in com com_vartype_convert type=%s'%type
    types = {
        '1':bin2int,
        '2':bin2float,
        '3':bin2str
    }
    ret = binstr    
    try:
        if size:
            ret = types[str(typecode)](binstr,size=size)
        else:
            ret = types[str(typecode)](binstr)
    except Exception,e:
        if verb:
        	print '\t',e,\
                    'An error occured while converting',\
                        'the binary string. binstr: %s, size: %s, typecode:%s'%(binstr,size,typecode)
    return ret
    

#GETTERS
def get_directory_header_scheme():
    '''returns an ordered list of tuples describing the scheme for the directory header'''
    return dir_header_scheme , scheme_size(dir_header_scheme)
def get_dataset_header_scheme():
    '''returns an ordered list of tuples describing the scheme for the dataset headers in the directory'''
    return dset_header_scheme ,scheme_size(dset_header_scheme)
def get_dataset_mandatory_header_scheme():
    '''returns an ordered list of tuples describing the scheme for the mandatory header beginning each dataset'''
    return dset_mandatory_header_scheme , scheme_size(dset_mandatory_header_scheme)

# gets the directory header
def get_dir_header(fp):
    ''' gets the directory header '''
    return parse_dir_head(fp)
# gets a dset body identified by name
def get_dset_binary(fp, dset_id, dset_header= None):
    ''' returns the binary of a given dset identified by that dset id
    '''
    if dset_header is None:
        dset_header = get_dset_header(fp,dset_id)
    size_of_dset = dset_header['size']
    block_num_of_start = dset_header['start']
    go_to_block(fp,block_num_of_start)
    # eat the header. we just want the body
    return fp.read(size_of_dset * block_size_bytes)
# gets a dset header identified by name
def get_dset_header(fp, dset_id, dir_head = None):
    ''' returns parsed dset header with given dset id
        if no dset with header is found throws exception.
    '''
    dset_heads = get_dset_headers(fp)
    try:
        dset_heads = [head for head in dset_heads if dset_id == head['id']]
        return dset_heads[0]
    except:
        raise Exception('dataset id %s was not found in this file'%dset_id)
# gets all dset headers
def get_dset_headers(fp, dir_header = None):
    ''' gets a list of all dset headers
    '''
    # initialize array to dave all dset headers
    dset_heads = []

    # we need to know how many dsets there are
    if dir_header is None:
        dir_header = parse_dir_head(fp)
    # go to the first dset header
    fp.seek(scheme_size(dir_header_scheme))
    for i in range(dir_header['number of datasets']):
        dset_heads.append(parse_dset_header(fp))
    return dset_heads
# gets a list of dset ids
def get_dset_ids(fp, dir_head = None):
    '''returns a list of dataset id's in order
    '''
    if not dir_head:
        dir_head = parse_dir_head(fp)
    dheadids = []
    for i in range(dir_head['number of datasets']):
        dheadids.append(parse_dset_header(fp)['id'])
    return dheadids

# calculates the size of a given header defintion.
def scheme_size(scheme):
    ''' calulates the size of a header block as determined from the deifintion in bytes
        requires the schema to follow the standard schema defn format.
        read parsedat.__doc__ for more 
    '''
    try:
        sz = 0
        for fld in scheme:
            if len(fld) is 3:
                sz += fld[1]
            elif len(fld) is 4:
                sz += fld[2]
    except TypeError:
        raise Exception('Scheme is not static. size cannot be determined.')
    return sz
# seeks to a specific block in the file
def go_to_block(fp,block):
    ''' goes to a specific block in the file
    '''
    # blocks are indexed by 1
    fp.seek((block -1) * block_size_bytes)
    # i.e. block 1 -> 0, block 2-> 512 etc...

def repgroup(fp, numreps, scheme, verb=True, raw = False):
	''' this function generates an array of dictionaries
		that is a repeating group defined by scheme. 
		it repeats numreps # of times 
		assumes fp is pointing to the start of the repeating group
	'''
	arr = []
	for i in range(numreps):
		if verb: print '--> new repgroup'
		arr.append(use_schema_definitions(fp, scheme, verb=verb, raw=raw))
	return arr


# SCHEMA DEFINITIONS
dir_header_scheme = [
        ('animal id', 12, str), # subject associated with file
        ('number of datasets', 4, bin2int), 
        ('number of blocks in directory', 4, bin2int), # size of directory
        ('unused1', 4, str),
        ('last updated', 8, str), # last time this file was updated. NOT USED.
        ('unused2',32, str)
        ]

dset_header_scheme =[
        ('name', 8, str),    # what schema the dataset uses. IMPORTANT
        ('size','dataset size in blocks', 4, bin2int),
        ('id','dataset id', 12, str),
        ('start','block number where dataset starts', 4, bin2int), # the block at which the dataset starts
        ('exp','Experiment type', 4, str) # Populin uses COM, the program running the experiment
        ]

dset_mandatory_header_scheme = [
        ('schema name',8, str), # this will determine how the dataset it organized
        ('dataset size in blocks',4, bin2int), # the number of blocks this dataset is
        ('animal id',12,str),   # should match the dir_header
        ('dataset id',12,str),  # should match dset_header
        ('date',8,str),         # probably at the beginning DDMMM-YY
        ('time',4,bin2int),     # in 10ths of a second since midnight
        ('experiment type code',4,str) # COM for POPULIN
        ]

scho18_scheme1 = [
    ('UDATA',   ' =0 No UET data, =1 Yes UET data'  ,4,bin2int   ),
    ('ADATA',   ' =0 No A/D data, =1 Yes A/D data'  ,4,bin2int   ),
    ('CMDATA',  ' =0 No CM data,  =1 Yes CM data'   ,4,bin2int   ),
    ('SP1CH',   ' Spikes UET channel number'        ,4,bin2int   ),
    ('STRTCH',  ' Start sync. UET channel number'   ,4,bin2int   ),
    ('TERMCH',  ' Terminate UET channel number'     ,4,bin2int   ),
    ('INWCH',   ' Enter Window UET channel number'  ,4,bin2int   ),
    ('REWCH',   ' Reward start UET channel number'  ,4,bin2int   ),
    ('ENDCH',   ' End Trial UET channel number'     ,4,bin2int   ),
    ('TBASE',   ' UET times base in seconds'        ,4,bin2float ),
    ('STFORM',  ' Status table format code'         ,4,bin2int   ),
    ('NUMPT',   ' No. of pointers in STATUS table'  ,4,bin2int   ),
    ('LSTAT',   ' Location of Status table'         ,4,bin2int   ),
    ('NSEQ',    ' No. of sequences in Status table' ,4,bin2int   ),
    ('RNSEED',  ' Seed used for random number generator'    ,4,bin2int   ),
    ('TGRACL1', ' Grace time for LED-1 ? (secs)'    ,4,bin2float ),
    ('TSPOTL1', ' Time to spot LED-1 ? (secs)'      ,4,bin2float ),
    ('TGRACL2', ' Grace time for LED-2 ? (secs)'    ,4,bin2float ),
    ('TSPOTL2', ' Time to spot LED-2 ? (secs)'      ,4,bin2float ),
    ('SPONTIM', ' Spontaneous time ? (secs)'        ,4,bin2float ),
    ('ISDREW',  ' Inter-seq delay after reward (secs)'  ,4,bin2float ),
    ('ISDNOREW',' Inter-seq delay after no-reward (secs'    ,4,bin2float ),
    ('ATTLOW',  ' Attenuator low value (dB)'        ,4,bin2float ),
    ('ATTHIGH', ' Attenuator High value (dB)'       ,4,bin2float ),
    ('ATTINC',  ' Attn. Step size (dB)'             ,4,bin2float ),
    ('LAPEND',  ' Location of "appended data" table',4,bin2int   ),
    ('SCAPEND', ' Schema name for appended data'    ,8,str       )
    ]

# this is a repeating group for fixed params
scho18_fxpar_head = [
    ('LFXPAR','Length of FXPAR RG (words)', 4, bin2int),
    ('NFXPAR','Number of fixed variables' , 4, bin2int)
    ]
scho18_fxpar_body = [
    ('FXVNAM','fixed variable name',8,str),
    ('FXVTYP','Fixed Var type 1=int,2=fp,3=char',2,bin2short),
    ('FXVLEN','length of variable in 4 byte words',2,bin2short),
    ('FXVVAL','value of variable', 'FXVLEN','FXVTYP')
	]

# this comes before subtask parameters
scho18_scheme2 = [
	('COMMENT', 60, str),
	('NSUBTASK','No. of sub-tasks in this data set', 4, bin2int)
	]

scho18_subtpar_head = [
	('LSUBTPAR','Length of SUBTPAR RG (words)',4,bin2int),
	('NSUBTPAR','Number of sub-task parameters',4,bin2int)
	]
scho18_subtpar_body = [
	('SUBTVNAM','Sub-Task Variable name',8,str),
	('SUBTVTYP','Sub-Task Var type 1=int,2=fp,3=char',2, bin2short),
	('SUBTVLEN','Length in 32-bit words',2, bin2short),
	('SUBTVVAL','Sub-Task variable value','SUBTVLEN','SUBTVTYP')
	]

scho18_scheme3 = [
	('AVOLC', 'Voltage conversion factor', 4, bin2float),
	('AVCC', 'Voltage Conversion Code', 4, bin2int),
	('ANBITS', 'No. of bits per sample 16/32', 4, bin2int),
	('NACH', 'No. of A/D channels', 4, bin2int)
	]

scho18_adch = [
	('ACHAN', 'A/D Channel number', 4, bin2int),
	('SRATE', 'Sampling rate (samples/sec)', 4, bin2float),
	('ASAMPT', 'Analog sampling time in secs', 4, bin2float),
	('NSAMP', 'Number of A/D samples', 4, bin2int)
	]


scho18_scheme4 = [
	('COILCODE', 'Coil calib. code (1,2,3 etc.)',4, bin2int),
	('NCOILCOF', 'Number of coil calib coefficients',4, bin2int),
	('NCOIL', 'Number of coils',4, bin2int)
	]

scho18_coilinf_head = [
	('COILPOS', 'Coil position (e.g. LEFTEAR,RITEAR)', 8, str),
	('ADCHX', 'A/D channel number for X-position', 4, bin2int),
	('ADCHY', 'A/D channel number for Y-position', 4, bin2int)
	]
scho18_coilinf_body = [
	('COFX','Coefficient for X-direction', 4, bin2float),
	('COFY','Coefficient for Y-direction', 4, bin2float)
	]

scho18_scheme5 = [
	('AVOLCCM', 'voltage conversion factor for CM', 4, bin2float),
	('AVCCCM', 'voltage conversion code for CM', 4, bin2int),
	('ANBITSCM', 'vits/sample for CM (16 or 32)', 4, bin2int),
	('ACHANCM', 'vhannel number for CM', 4, bin2int),
	('NUMCM', 'vumber of CM recordings per trial', 4, bin2int),
	('NUMLED', 'votal number of LEDs', 4, bin2int)
	]

scho18_ledpos = [
	('LEDPAZIM','LED azimuth position (-180 to +180)',4,bin2float),
	('LEDPELEV','LED elevation pos. (-90 to +90)',4,bin2float)
	]

scho18_scheme6 = [
	('NUMSPK','Number of Speakers', 4, bin2int)
	]

scho18_spkpos = [
	('SPKPAZIM','Speaker azimuth pos. (-180 to +180)',4,bin2float),
	('SPKPELEV','Speaker elevation pos. (-90 to +90)',4,bin2float)
	]

scho18_scheme7 = [
	('LDUMMY',4, bin2int),
	('DUMMY','LDUMMY',str)
	]

scho18_DERV = [
    ('DERVNAM', 8, bin2str),
    ('DERVTYP', 2, bin2short),
    ('DERVLEN', 2, bin2short),
    ('DERVAL', 'DERVLEN', 'DERVTYP')
    ]

block_size_bytes = 512
loc_first_dset_header = 64
loc_start_dir_headr = 0


# opens a dat file
def open_dat(path = 'gordon_505.dat'):
    '''Open a datafile. defaults to the one here.'''
    return open(path,'rb')


# manipulates the schemes defined above. 
def use_schema_definitions(fp, scheme, comments = False, verb = True, raw=False):
    '''
        we have a standard form of schemes in this module. 
        They can be used to parse the files assuming the
        file pointer is in the right place. 
        this abstracts that.  

        
        size can be either an integer or a field earlier in the scheme 
        contianing a number 

        parsing function can be either a function name,
        integer (1=int,2=fp,3=string), or a field earlier in the scheme pointing 
        to a type

        This supports these standard forms for schema definitions
        1.
            [(field name, size, parsing function),
            ...]
        2.
            [(field name, comment, size, parsing function),
            ...]

            schemas can mix and match these standard forms.
    '''
    startpoint = None
    parsed_dict = {}
    conversion_type = None # if the type is dependant on a field we must save it temp.
    for field in scheme:
        desc = None # use these in the finally
        val = None # use these in the finally 
        buf = None # to read the data in
        try:
        	try:
	            if len(field) is 3:
	                desc = field[0]
	                size = field[1]
	                parse = field[2]
	                comment = desc
	            elif len(field) is 4:
	                desc = field[0]
	                comment = field[1]
	                size = field[2]
	                parse = field[3]
	
	            # check that size is not a field name
	            if type(size) is str: 
	                # if size sia a sring, assuming it to be a field name. 
	                #getting value at that field.
	                field_holding_size = size
	                try:
	                    if comments:
	                        size = parsed_dict[field_holding_size]['value']
	                    else:
	                        size = parsed_dict[field_holding_size] 
	                    
	                    size = words2bytes(size) # convert the size from 32 bit words to bytes
	                except Exception:
	                    raise Exception('Field %s not defined before %s was to be parsed.'\
	                                    %(field_holding_size, desc))
	                
	            # Read in the size 
	            buf = None
	            try:
	                startpoint = fp.tell()
                        buf = fp.read(size)
	            except Exception:
	                raise Exception('Error encountered while reading the scheme! size =%s'%str(size))        
	            
	            # check if type is defiend with an integer (1=int,2=fp,3=string)
	            if type(parse) is int:
	                conversion_type = parse # save the type passed in via parse
	                parse = lambda x: com_vartype_convert(x,conversion_type,size=size, verb=verb) # set parse to a callable function
	            
	            # check if type is defined with another field
	            elif type(parse) is str:
	                field_holding_type = parse # save the field
	                try:
	                    if comments: # then it's a dict of dicts, we want the value
	                        conversion_type = parsed_dict[field_holding_type]['value']
	                    else:
	                        conversion_type = parsed_dict[field_holding_type]
	                    parse = lambda x: com_vartype_convert(x,conversion_type, size=size,verb=verb)
	                except Exception:
	                    # the field referanced does not exist yet!
	                    raise Exception('Field %s not defined before %s was to be parsed.'\
	                                    %(field_holding_type, desc))
	                
	            # if we want to include comments then returns a dict of dicts
	            # else, just return key_values
	            val = parse(buf)
	            if not comments:
	                parsed_dict[desc] = val
	            else:
	                parsed_dict[desc] = {}
	                parsed_dict[desc]['comment'] = comment
	                parsed_dict[desc]['value'] = val
	        except Exception,e:     # parse can be uncallable, if it shouldn't be parsed.
	            if verb: print '\t', 'error while parsing field: %s. error-> %s'%(str(field), str(e))
	            # if it fails. then return the raw, and continue.
	            # if desc or size are not defined. throw error
	            val= buf
	            parsed_dict[desc] = val 
        finally:
        	if verb: print startpoint/4, desc, val, 'EOL'
        	if raw: parsed_dict['raw'] = buf    
    return parsed_dict

# parses the directory header
def parse_dir_head(fp, comments = False):
    ''' parses the header of the directory.
        the dat file is composed of blocks 512 bytes.
        the first set of blocks are called the directory,
        which catalogs what datasets are in the whole.

        the directory has a header. this parses that.
        using the dir_header_scheme defined in this module

        This does not expect to be at the beginning of the file.
    '''
    fp.seek(0) # the directory is always at the beginning.
    
    return use_schema_definitions(fp, dir_header_scheme, comments = comments)
# parses a dset header
def parse_dset_header(fp, comments = False):
    ''' this reads in and parses the dataset headers.
        this expects the fp to be pointing at the start of the header.
        it also expects the header to follow th schema defined in this module.
    '''
    return use_schema_definitions(fp, dset_header_scheme, comments = comments)
 # parse a mandatory header for a dataset
def parse_mandatory_header(fp, comments = False):
    ''' This reads in and parses the mandatory dataset header 
        at the beginning of each dataset. 
        it expects the file_pointer to be pointing at the begginning
        of the header
    '''
    return use_schema_definitions(fp, dset_mandatory_header_scheme, comments = comments)
# parses a single dataset given it's header
def parse_dset_body(fp, comments = False):
    ''' parses the body of a scho18_scheme dset body. 
        expects to be pointing at the beginning of the dset body
    '''
    return use_schema_definitions(fp, scho18_scheme1, comments = comments) 



def pprint(struct):
	print '======'
	for i in struct:
		print i
	print '-----'
def read_vec(fp, elem_parser, size, name=None):
    length = bin2int(fp.read(4))
    if name: print('%s elements in vector %s'%(length,name))
    else: print('%s elements in this vector'%length)
    for i in range(length):
        x = elem_parser(fp.read(size))
        if i < 20:
            print(x)
    if (length*size)%4 != 0:
        #print('length*size = %s. which is not 0 mod 4. reading in 1 more'%length*size)
        fp.read(size)

def read_list(fp, elem_parser, size,  length, name =None):
    if name: print('in list __%s__ there is:'%name)
    for i in range(length):
        x = elem_parser(fp.read(size))
        if i < 20:
           print(i,x)
    if (length*size)%4 !=0:
        #print('length*size = %s. which is not 0 mod 4. reading in 1 more'%(length*size))
        fp.read(size)

def read_vec_list(fp, vec_elem_parser,size,length,name=None):
    if name: print('in list of vectors __%s__ there are %s vecotrs:'%(name,length))
    if not name: name = ''
    for i in range(length):
        read_vec(fp,vec_elem_parser, size, name = name + str(i))
def read_stb():
    maps = {
        1 : bin2int,
        2 : bin2float,
        3 : str
    }
    name = str(f.read(8))
    typ = bin2short(f.read(2))
    length = bin2short(f.read(2))
    if typ not in maps:
        return name,typ,length,None
    if typ != 3 and length != 1:
        return name,typ,length,None
    else:
        return name,typ,length, maps[typ](f.read(length*4))


f = open_dat('swigert_391.dat')#'RL_hu_test.dat')#  open the file )#
dset_ids = get_dset_ids(f) # get a list of datasets
dir_head = get_dir_header(f) # get the directory header
dsets = get_dset_headers(f) # get the dset headers

go_to_block(f,dsets[0]['start']) # go to the start of dset 1
dset1_manhead = parse_mandatory_header(f) # parse the mandatory header (32 bytes)
dset1_body1 = parse_dset_body(f, comments = True)  # parse scho18


fxvarh = use_schema_definitions(f,scho18_fxpar_head)
fxvarbs = repgroup(f,fxvarh['NFXPAR'], scho18_fxpar_body, raw=True)

dset1_body2 = use_schema_definitions(f,scho18_scheme2)

subtpars = []
safe = f.tell()

for i in range(dset1_body2['NSUBTASK']):
	subtpar_head = use_schema_definitions(f,scho18_subtpar_head)
	subtpar_body = repgroup(f,subtpar_head['NSUBTPAR'], scho18_subtpar_body, verb=True)
	subtpars.append({'head':subtpar_head,'body':subtpar_body})


dset1_body3 = use_schema_definitions(f, scho18_scheme3)

scho18_adch = repgroup(f, dset1_body3['NACH'], scho18_adch)

dset1_body4 = use_schema_definitions(f, scho18_scheme4)

coilinfs = []
for j in range(dset1_body4['NCOIL']):
	coilinf = {}
	coilinf['head'] = use_schema_definitions(f,scho18_coilinf_head)
	coilinf['body'] = repgroup(f,dset1_body4['NCOILCOF'], scho18_coilinf_body)
	coilinfs.append(coilinf)

dset1_body5 = use_schema_definitions(f,scho18_scheme5)
scho18_ledpos = repgroup(f,dset1_body5['NUMLED'], scho18_ledpos)

dset1_body6 = use_schema_definitions(f,scho18_scheme6)
scho18_spkpos = repgroup(f,dset1_body6['NUMSPK'],scho18_spkpos, verb=True)


bfd = f.tell()
dset1_body7 = use_schema_definitions(f,scho18_scheme7)
afd = f.tell()

safe = f.tell()
# we are currently pointing to the unknown vector stuff. 
#	01  DATA TYPE RG OCCURS NSEQ TIMES
#	    02  TSDATA TYPE VECTOR INTEGER                  /* spike time data */
#	    02  ANDATA TYPE VECTOR I*2 OCCURS NACH TIMES    /* sampled analog data */
#	    02  CMDATA TYPE VECTOR I*2 OCCURS NUMCM TIMES   /* sampled CM data */
#	    02  NDERV                                       /* no. of derived vars */
#	    02  DERV TYPE RG OCCURS NDERV TIMES
#	        03 DERVNAM TYPE STRING 8                    /* name of derived var */
#	        03 DERVTYP TYPE I*2                         /* variable type */
#	        03 DERVLEN TYPE I*2                         /* length in 32 bit wrds */
#	        03 DERVAL LENGTH DERVLEN                    /* val of derived var */
print 'NSEQ', dset1_body1['NSEQ']
print 'NACH', dset1_body3['NACH']
print 'NUMCM', dset1_body5['NUMCM']





for j in range(dset1_body1['NSEQ']['value']):
    print('data entry %s of %s. loc in words = %s'%(j+1,dset1_body1['NSEQ']['value'], f.tell()/4))
    read_vec(f,bin2hex,4,name='TSDATA') 
    print('an data is starting on word %s'%str(f.tell()/4))
    read_vec_list(f,bin2short,2,dset1_body3['NACH'],name='ANDATA')
    read_vec_list(f,bin2short,2,dset1_body5['NUMCM'],name='CMDATA')
    nderv = bin2int(f.read(4))
    print('%s dervs follow'%nderv)
    for i in range(nderv):
        print('nderv is 0. not caring right now')

for j in range(dset1_body1['NSEQ']['value']):
    print '\t STAT TABLE %s'%(j+1)
    nvstat = bin2int(f.read(4))
    print('%s vars in stat table %s'%(nvstat, j+1))
    print('loc of stat table %s is %s'%(j+1, f.tell()))

    for i in range(nvstat):
        n,t,l,v = read_stb()
        if v is None:
            print('Error processing %s(%s)[%s]'%(n,t,l))
            break
        print('%s (%s)[%s] = %s'%(n,t,l,v))    
    for k in range(dset1_body1['NUMPT']['value']):
        print('ptr %s is %s'%(k,bin2int(f.read(4))))
    
