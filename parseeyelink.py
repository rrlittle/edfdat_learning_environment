''' 
    Some terminology.

    trial... Eyelink defines a trial to be one recortding period
    trial set ... a collection of events during a trial
                    basically each event is one line in the asc file
                    during a trial period
'''



from os.path import join
import re
import logging
import struct as s
import binascii as ba
# import matplotlib.pyplot as plt

def open_edf(path=['temp.txt']):
    ''' opens a asci version of an edf file and returns the pointer
        as readonly. defaults to memory.txt for fun... :)
        '''
    return open(join(*path), 'r')

def get_trial_sets(fp,reset=True):
    ''' returns a list of lists.
        where each sublist represents all the
        statements from that trial

        if reset is True starts from byte 0 else 
        starts from current location
        returns
        [   [trial1.1, trial1.2,...],
            ...
        ]

        the first entry is the header. if there are no lines
        it will be empty. 

        moves fp to 0
        '''
    trials = [] # the initial list is the header
    if reset: fp.seek(0) # go to the head

    for statement in fp: # go line by line
        if len(trials) == 0: trials.append([]) 
        # ensure theres one there. the first 'trial' is the header
        if emptystr(statement): continue
        if matches_trial_start(statement):
           trials.append([])
        trials[-1].append(statement)
    fp.seek(0)
    return trials

def emptystr(string):
    ''' returns T/F if string is empty or no
        '''
    return len(re.findall(r"^[\s]*$", string)) > 0 # look for empty lines    

def matches_trial_start(statement):
    ''' determines if a statement (or line from an eyelink datafile )
        is the start of a new trial.
        basically if it matches this pattern
        MSG\t<TIMESTAMP> TIRALID <TIRALID>

        that's what we're calling the start of the trial. 
        ''' 
    return len(re.findall(r"^MSG\t\d+ TRIALID \d+$", statement)) > 0

def msg_parser(trial_str, logger=logging, *args, **kwargs):
    ''' this parses a trial string containing a message
        returns a dict containing
        {
            'type':MSG
            'val':string
            'timestamp':float. objective camera time. always increasing
        }
        '''
    split_tr = re.split(r'\s+', trial_str)
    ret = {}
    try:
        ret['type'] = split_tr[0]
        ret['timestamp'] = float(split_tr[1])
        ret['value'] = '\t'.join(split_tr[2:])
    except (IndexError, ValueError):
        logger.error('Attempting to parse message string ' + 
            '(%s). Not formatted as expected')%trial_str
        return None
    return ret

def sample_parser(trial_str, 
    monocle = True, 
    velocity = False, 
    resolution = False,
    remote = True,
    CR = True,
    logger=logging,
    *args,
    **kwargs):
    ''' this parses a trial string containing a sample
        returns a dict with keys representing fields in the
        sample.

        the fields present are determined by the length and the flags
        e.g.
        string :    1 2 3 4 5 ... 6 7 8 .............
        results in ->
            timestamp = 1
            xpos = 2
            ypos = 3
            pupil = 4
            # note 5 is unkown and is being ignored
            monocle_errs = ...
            xpos_camera = 6
            ypos_camera = 7
            distance = 8
            remote_errs = .............
        '''
    def deal_with_missing_data(data):
        ''' this returns a float value always. for this data point.
            if the data is missing it will be 9998. else the correct value'''
        if data == '.': return float(9998)
        else: return float(data)

    split_tr = re.split(r'\s+', trial_str)
    logger.debug('In parse sample. split string = %s'%split_tr)
    tr_iter = split_tr.__iter__()
    ret = {}
    try:
        ret['type'] = 'sample'
        ret['timestamp']    = float(tr_iter.next())
        if monocle:
            ret['xpos']     = deal_with_missing_data(tr_iter.next())
            ret['ypos']     = deal_with_missing_data(tr_iter.next())
            ret['pupil']    = float(tr_iter.next())

            if velocity: ####################################################### this needs to be tested! I'm not sure how adding resolution/velocity will affect column structure 
                ret['xvel']     = float(tr_iter.next())
                ret['yvel']     = float(tr_iter.next())
            if resolution:
                ret['xres']     = float(tr_iter.next())
                ret['yres']     = float(tr_iter.next())

            tr_iter.next() # there is an unknown quantity. 
            if CR: ret['CR_err']     = tr_iter.next()

            if remote: # then there are some more columns
                ret['xpos_camera']  = float(tr_iter.next())
                ret['ypos_camera']  = float(tr_iter.next())
                ret['distance']     = float(tr_iter.next())
                ret['remote_errs']  = tr_iter.next()
        else: # not monoclular mode different structure. page 120 in user manual
            raise NotImplementedError('binocular mode has not been written yet'
                                        + ' dont worry. its not hard')
        
    except(StopIteration, ValueError), e:
        logger.error('Trial str (%s) did not meet expectations based on '%trial_str
                    + ' flags (mono:%s, vel:%s, res: %s, remote:%s, CR:%s).'%(monocle, 
                        velocity, resolution, remote,CR)
                    + ' Error: %s' %str(e))
        return None 
    return ret

def parse_trial(trial_str, logger=logging, **kwargs):
    ''' This takes a trial string
        i.e. 
        {TIMESTAMP}{X}{Y}{ps}{UNKOWN}{MONOCLE ERRORS CHARS}\
                            {X camera}{y camera}{distance}{remote ERRORS}
        returns a dictionary or None if it can be ignored
        dict will defintly have type = sample\message
        if type= sample
            it will have all the keys as a sample line

        if type = MSG
            it will have key string & TIMESTAMP.

        kwargs can contain flags for parsing. things like 
        CR  # Corneal reflection mode
        remote # remote mode
        velocity, resolution etc. flags set when creating asci file. 

        if they are not provided defaults will be used

        '''
    def err(): # we call the same error message a few times here. 
        logger.debug('in parse_trial:{%s} is neither a sample nor a MSG.'%trial_str)
        return None

    timestamp = None
    matchfloat = r"[-]?\d*\.*\d+" # e.g. -13.5
    matchfirstval = r"^[0-9a-zA-Z]*" # e.g. MSG\TIMESTAMP\EFIX\SSACC
    stype = None
    try:
        stype = re.findall(matchfirstval,trial_str)[0]
    except IndexError: return err() # not found
        
    typeparsers = {
        'MSG':msg_parser,
        'sample':sample_parser
    }
     
    try: # see if it's a sample. everything else begins with a string of type
        timestamp = float(stype)
        stype = 'sample'
    except ValueError: pass # it's not a sample
            
    if stype in typeparsers:
        try:
            return typeparsers[stype](trial_str, logger=logger, **kwargs) ########## NEED TO UPDATE monocle should reflect whether theres a monocle or not as set by global variable or something...
        except NotImplementedError: return err() 
    else:
        return err()

def parse_trial_set(tr_list, 
    org = 'single',
    logger=logging,
    **kwargs):
    ''' this take a list of trial strings and parses them using parse trial.
        returns a list of dicts.. pretty simple really.

        Oh! this drops trials that aren't either message or sample

        also you notice the org kwarg. that accepts single, multi 
        single returns an array with both samples and messages
        multi returns a dict with two array based on type.
            e.g. {msg:[....], samples:[...]}

        kwargs should be flags such as monocle
        '''
    ret = None
    if org == 'single': ret = []
    else: ret = {} 
    for trial in tr_list:
        parsed_trial = parse_trial(trial, logger=logger, **kwargs)
        # if this fails it will just print error message to stdout
        # and return None
        if parsed_trial is None: 
            if org != 'single': 
                if 'dropped' not in ret: ret['dropped'] = []
                ret['dropped'].append(trial)
            continue
        if org == 'single': ret.append(parsed_trial)
        else:
            tr_type = parsed_trial['type'] 
            if tr_type not in ret: ret[tr_type] = []
            ret[tr_type].append(parsed_trial)
    return ret 

def message_in_map(message, maps):
    ''' message is any string, but should probably be the value 
        from msg_parser. 

        this will search for a match within the map.
        where map is a list of dicts. each dict should have
        key, match, value OR valuestr

        this will apply re.match to each of the dicts in map 
        if that returns a match object a dictionary is created
        with key and value. 
        key comes from map.key
        and value comes from either map.value or map.valuestr

        if both map has both value and valuestr badmapError is raised

        args
        ----
        messge : '-------'
        maps : 
        [   {   'key':'----', 'match':r'----', value:999},
            {   'key':'----', 'match':r'----', 
                'valuestr': lambda x: re.findall(r'---',x)[0]   },
            ...
        ]

        returns
        -----
        [   {'key':'---', 'value':----},
            ....
        ]
        '''
    class badmapError(Exception):
        """thrown if maps is poorly defined"""
        pass

    matches = []
    for m in maps:
        if 'match' not in m: 
            raise badmapError('match should be a key within %s'%m)
        
        if re.match(m['match'], message):
            match = {}
            if 'key' not in m: 
                raise badmapError(('key should be a '
                                    'key within %s')%m)
            match['key'] = m['key']
            if 'value' in m and 'valuestr' in m: 
                raise badmapError(('only one of value or valuestr '
                                    'should be ky in %s')%m)
            if 'value' in m: match['value'] = m['value']
            elif 'valuestr' in m: match['value'] = m['valuestr'](message)
            else: match['value'] = None
            matches.append(match)
    return matches

def parse_messages(message_list, message_mapping, logger=logging):
    ''' this takes a list of message dicts and saves them with useful keys
        this is obviously very dependent on the sturucture of the experiment
            i.e. what messages are produced
        so we have to define a mapping and some regex.
        For now We will define that here and make sepereate functions 
        for each experiment and just use those based on our knowledge. 

        but... in the future we'll want something better

        keys of all returned dicts: 
            value: the value of the thing, 
            timestamp: when the thing was recorded relative to start of trial, 
            type: event/info/define your own in map. 
                if type not found in match. defaults to info. 
        '''
    ret = {}
    for m in message_list:
        matches= message_in_map(m['value'], message_mapping)
        # returns either None or {'key':----, 'value':-----}
        if len(matches) > 1: 
            logger.error('%d matches for message %s. continuing'%(len(matches),m))
            for match in matches:
                logger.error('%s'%match)
            continue
        elif len(matches) == 0:
            logger.debug('No matches for message %s. continuing'%m)
            continue 
        match = matches[0]
        if match:
            message_dict = {   'value':match['value'],
                                'timestamp': m['timestamp']
                                }
            if 'type' in match: message_dict['type'] = match['type']
            else: message_dict['type'] = 'info' # defaults to event
            ret[match['key']] = message_dict
    return ret

class write_var_error(Exception):
    ''' error in write_var function''' 
    pass

def parse_edf(edfpath, exp_msgs, logger=logging):
    ''' this takes a file pointer to a edf file and 
        parses it. requires a referance to what type experiment it is. 
        exp_msgs should be one of the experiment maps for parsing messages
        '''
    f = open_edf(edfpath)
    sets = get_trial_sets(f) # breaks edf into trial sets.
    hdr = sets[0] # grab the header as it's special
    sets = sets[1:] # grab the trial sets. which have not been parsed yet
    parsed_trs = [] # prepare to save parsed trials
    for i,tr_set in enumerate(sets): # parse each trial
        logger.info('processing trial %s of %s'%(i+1,len(sets)))
        parsed_tr = parse_trial_set(tr_set, # parse the trial. i.e. break into 
                    org='multi',            # samples, messages, 
                    logger=logger)          # and unkown stuff
        
        dropped = []
        if 'dropped' in parsed_tr: dropped = parsed_tr['dropped']
        samples = []
        if 'sample' in parsed_tr: samples = parsed_tr['sample']
        messages = []
        if 'MSG' in parsed_tr: messages = parsed_tr['MSG']

        parsed_messages = parse_messages(messages, exp_msgs, logger=logger)
        msgs_by_typ = {}
        for pmkey,pmdic in parsed_messages.iteritems():
            typ = pmdic['type'] 
            if typ not in msgs_by_typ: msgs_by_typ[typ] = {}
            msgs_by_typ[typ][pmkey] = pmdic

        logger.debug('\tfound %s samples'%len(samples))
        logger.debug('\tfound %s messages with %s types'%(\
            len(parsed_messages),len(msgs_by_typ)))
        logger.debug('\tcould not parse %s samples'%len(dropped)) 
        tr_dict = {
            'samples':samples,
            'dropped': dropped,
            'messages': msgs_by_typ
        }
        parsed_trs.append(tr_dict)

    return {'header':hdr,
            'trials':parsed_trs}


block_size_bytes = 512
def uet2bin(channel, timestamp, logger=logging):
    ''' constructs an integer representing a uet event
        which is a 4 byte integer. that
        time = int & 0x00ffffff
        channel = (int & 1f000000) >> 24
        i.e. 0xttttttcc t = time bit, c = channel bit'''
    val = int2bin(timestamp,logger=logger)
    val = val[:-1] # replace the last 2 characters with channel
    val += short2bin(channel, logger=logger)[0:1]
    logger.critical(bin2hex(val))
    return val
def str2bin(value, size, logger=logging):
    ''' returns size byte bin str'''
    return (value + " "*size)[0:size]
def int2bin(value, logger=logging):
    ''' returns 4 byte bin str '''
    return s.pack('i', value)
def float2bin(value, logger=logging):
    ''' returns 4 byte bin str '''
    return s.pack('f', value) 
def short2bin(value, logger=logging):
    ''' returns 2 byte bin str '''
    return s.pack('h',value)
def vector2bin(arr, elem_typ,logger=logging):
    maps = {
        'int':int2bin,
        'float':float2bin,
        'short':short2bin,
        'bin':lambda x: x
    }
    logger.debug('passed in arr of len %s with type %s'%(len(arr), elem_typ))
    if elem_typ not in maps: 
        raise ValueError('%s not a supported for vectors'%elem_typ)
    elem_parser = maps[elem_typ]
    num_elems = len(arr)
    binstr = int2bin(num_elems)
    for elem in arr:
        binstr += elem_parser(elem)
    if elem_typ == 'short' and num_elems%2 != 0:
        logger.debug('padding binstr with 2 bytes')
        binstr += str2bin('',2) # add 2 bytes to maintain 4 byte alignment
    return binstr
def list2bin(arr,elem_typ,vec_elem_typ=None,logger=logging):
    ''' this creates a string representing a list of values.
        this differs from vectors because the length is defined in the 
        schema and not by an integer preceding the struct. 
        array is a list of values. 
        it can be:
            float,
            int
            short
            or vector
        if vector is supplied a vector_type is requied and can be any type
        from vector2bin. 
        '''
    maps = {
        'int':(int2bin,4),
        'float':(float2bin,4),
        'short':(short2bin,2),
        'vector':(lambda x,logger=logging: vector2bin(x,vec_elem_typ,logger=logger),)
    }
    if elem_typ not in maps: 
        ValueError('%s not a supported for lists'%elem_typ)
    if elem_typ == 'vector' and vec_elem_typ is None:
        raise ValueError('if elem_typ is vector vec_elem_typ is required')
    binstr = ''
    for val in arr:
        binstr += maps[elem_typ][0](val,logger=logger)
    if elem_typ == 'short' and maps[elem_typ][1]%2 != 0:
        binstr += str2bin('',2) # add anothe 2 bytes to align the data
    return binstr 
def go_to_block(fp, block_to_go_to):
    ''' moves the fp to point to byte 0 of block provided. indexed by 1'''
    fp.seek((block_to_go_to - 1) * block_size_bytes) 
def write_var(fp, name, typ, length, val, name_len=8, logger=logging):
    ''' used to simplify writing varibales like fixed param 
        vars and subtpars
        They always have the same schema. 
        - name (str, 8 chars)
        - typ (short, 1 = int, 2 = float, 3 = str)
        - len (short) length in 4 byte words
        - value pass in binstr, this just writes 
        '''
    if typ not in [1,2,3]: 
        raise write_var_error('typ %s provided invalid'%typ)
    fp.write(str2bin(name,name_len))
    fp.write(short2bin(typ))
    fp.write(short2bin(length))
    fp.write(val)

def bin2float(binstr, logger=logging): # converts a binary to a float
    ''' converts a binary string to a float'''
    return s.unpack('f',binstr)[0]
def bin2int(binstr, logger=logging): # converts a binary string to an int
    ''' converts a binary string to integer'''
    return s.unpack('i',binstr)[0] # normally returns a tuple. we want an int
def bin2str(binstr, logger=logging): # converts a binary string to a str
    ''' converts binary string to python string '''
    return str(binstr) # python defaults to converting bin data to ascii   
def bin2short(binstr, logger=logging): # converts a binary string to a short
    '''converts a binary string to a short'''
    return s.unpack('h',binstr)[0]
def bin2hex(binstr, logger=logging): # converts a binary string to hex string for printing
    '''usees binascii.hexlify to convert to string'''
    return ba.hexlify(binstr)
def read_vec(fp, elem_parser, size, logger=logging):
    length = bin2int(fp.read(4))
    ret = []
    for i in range(length):
        x = elem_parser(fp.read(size))
        ret.append(x)
    if (length*size)%4 != 0:
        fp.read(size)
    return ret
def read_list(fp, elem_parser, size,  length, logger=logging):
    ret = []
    for i in range(length):
        x = elem_parser(fp.read(size))
        ret.append(x)
    if (length*size)%4 !=0:
        fp.read(size)
    return ret
def read_vec_list(fp, vec_elem_parser,size,length, logger=logging):
    ret = []
    for i in range(length):
        x = read_vec(fp,vec_elem_parser, size)
        ret.append(x)
    return ret

#--------------------- task specific


temp_dis_map = [
    {   'key':'example',
        'match':r'this will probably never match anything',
        'valuestr': lambda x: float(re.findall(r'^[\W\w]*%d+',x)[0]),
        },
    {   'key':'TRIALID',
        'match':r"^TRIALID\t\d+\t$",
        'valuestr': lambda x: float(re.findall(r'\d+',x)[0])
        },
    {   'key':'success flag',
        'match':r'^!CMD\t\d+\techo\t',
        'valuestr': lambda x : bool(re.findall(r'(?!\t)\w+(?:\t$)', x))
        },
    {
        'match':r'^!CMD\t\d+',
        'valuestr': lambda x : re.findall(r'SUCCESS', x)
        },
    {   'key':'experiment type',
        'match': r'^!V\tTRIAL_VAR\tdebug',
        'valuestr': lambda x: re.findall(r'(?!\t)\w+(?:\t$)', x)[0]
        },
    {   'key':'ss image',
        'match': r'^!V\tTRIAL_VAR\tss_image',
        'valuestr': lambda x: re.findall(r'airplane\d+\.png',x)[0]
        },
    {   'key':'ss reward',
        'match':r'!V\tTRIAL_VAR\tss_reward',
        'valuestr': lambda x: re.findall(r'\d+',x)[0]
        },
    {   'key':'ll delay',
        'match':r'!V\tTRIAL_VAR\tll_delay',
        'valuestr': lambda x: re.findall(r'\d+',x)[0]
        },
    {   'key':'ll y pos',
        'match':r'!V\tTRIAL_VAR\tll_y_pos',
        'valuestr': lambda x: re.findall(r'\d+',x)[0]
        },
    {   'key':'ll reward',
        'match':r'!V\tTRIAL_VAR\tLL_REWARD',
        'valuestr': lambda x: re.findall(r'\d+',x)[0]
        },
    {   'key':'ss delay',
        'match':r'!V\tTRIAL_VAR\tSS_DELAY',
        'valuestr': lambda x: re.findall(r'\d+',x)[0]
        },
    {   'key':'ss y pos',
        'match':r'!V\tTRIAL_VAR\tSS_Y_POS',
        'valuestr': lambda x: re.findall(r'\d+',x)[0]
        },
    {   'key':'score total',
        'match':r'!V\tTRIAL_VAR\tSCORE_TOTAL',
        'valuestr': lambda x: re.findall(r'\d+',x)[0]
        },
    {   'key':'fixation size',
        'match':r'!V\tTRIAL_VAR\tFIXATION_SIZE',
        'valuestr': lambda x: re.findall(r'\d+',x)[0]
        },
    {   'key':'fixation ia size',
        'match':r'!V\tTRIAL_VAR\tFIXATION_IA_SIZE',
        'valuestr': lambda x: re.findall(r'\d+',x)[0]
        },
    {   'key':'airplane width',
        'match':r'!V\tTRIAL_VAR\tairplane_width',
        'valuestr': lambda x: re.findall(r'\d+',x)[0]
        },
    {   'key':'airplane height',
        'match':r'!V\tTRIAL_VAR\tairplane_height',
        'valuestr': lambda x: re.findall(r'\d+',x)[0]
        },
    {   'key':'fixation duration',
        'match':r'!V\tTRIAL_VAR\tFIXATION_DURATION',
        'valuestr': lambda x: re.findall(r'\d+',x)[0]
        },
    {   'key':'ll x pos',
        'match':r'!V\tTRIAL_VAR\tLL_X_POS',
        'valuestr': lambda x: re.findall(r'\d+',x)[0]
        },
    {   'key':'ss x pos',
        'match':r'!V\tTRIAL_VAR\tSS_X_POS',
        'valuestr': lambda x: re.findall(r'\d+',x)[0]
        },
    {   'key':'ll airplane',
        'match':r'!V\tTRIAL_VAR\tLL_AIRPLANE',
        'valuestr': lambda x: re.findall(r'airplane\d+\.png',x)[0]
        },
    {   'key':'current block',
        'match':r'!V\tTRIAL_VAR\tCURRENT_BLOCK',
        'valuestr': lambda x: re.findall(r'\d+',x)[0]
        },
    {   'key':'training',
        'match':r'!V\tTRIAL_VAR\tTRAINING',
        'valuestr': lambda x: re.findall(r'(?!t)\w+(?:\t$)',x)[0] # grabs the last word
        },
    {   'key':'image chosen',
        'match':r'!V\tTRIAL_VAR\timage_chosen',
        'valuestr': lambda x: re.findall(r'airplane\d+\.png',x)[0]
        }        
]
color_map = [
    {   'key':'trial id',
        'match':r"^TRIALID",
        'valuestr': lambda x: float(re.findall(r'\d+',x)[0])
        },
    {   'key':'display fixation',
        'match':r'-?\d+\tDisplayFixation\t'
        },
    {   'key':'display target',
        'match':r"-?\d+\tDisplayTarget"
        },
    {   'key':'task type and trial end sentinel',
        'match':r"!V\tTRIAL_VAR\ttype",
        'valuestr': lambda x: re.findall(r'(?!\t)\w+(?:\t$)',x)[0] 
        },
    {   'key':'trial max count',
        'match':r"!V\tTRIAL_VAR\tTRIAL_MAX_COUNT",
        'valuestr': lambda x: re.findall(r'\d+',x)[0] 
        },
    {   'key':'task',
        'match':r"!V\tTRIAL_VAR\tTASK\t",
        'valuestr': lambda x: re.findall(r'(?!\t)\w+(?:\t$)',x)[0] 
        },
    {   'key':'block count',
        'match':r"!V\tTRIAL_VAR\tBLOCK_COUNT",
        'valuestr': lambda x: re.findall(r'\d+',x)[0] 
        },
    {   'key':'fixation delay',
        'match':r"!V\tTRIAL_VAR\tFixDelay\t",
        'valuestr': lambda x: int(re.findall(r'\d+',x)[0])
        },
    {   'key':'trial ok',
        'match':r"!V\tTRIAL_VAR\tTrial_OK\t",
        'valuestr': lambda x: int(re.findall(r'\d+',x)[0]) # 1 = ok, 0 = no
        },
    {   'key':'trial type',
        'match':r"!V\tTRIAL_VAR\tTRIAL_TYPE\t",
        'valuestr': lambda x: int(re.findall(r'\d',x)[0]) # 1 = right, 0 = left 
        },
    {   'key':'fixation error',
        'match':r"!V\tTRIAL_VAR\tfixation_error\t",
        'valuestr': lambda x: re.findall(r'(?!\t)\w+(?:\t$)',x) != 'False' 
        },
    {   'key':'target positions',
        'match':r"!V\tTRIAL_VAR\ttargetpositions\t",
        'valuestr': lambda x: re.findall(r'\[\(\d+,\t\d+\),\t\(\d+,\t\d+\)\](?:\t$)',x)[0] 
        },
    {   'key': 'break fixation event',
        'match':r'-?\d+\tERROR_BREAK_FIXATION\t'
        },
    {   'key':'success flag',
        'match':r'^!CMD\t\d+\techo',
        'valuestr': lambda x : re.findall(r'(?!\t)\w+(?:\t$)', x)
        }  
]
memory_map = [
    {   'key':'trial id',
        'match':r"^TRIALID",
        'valuestr': lambda x: float(re.findall(r'\d+',x)[0])
    },
    {   'key':'success flag',
        'match':r'^!CMD\t\d+',
        'valuestr': lambda x : re.findall(r'(?!\t)\w+(?:\t$)', x)
    },
    {   'key':'experiment type',
        'match': r'^!V\tTRIAL_VAR\tdebug',
        'valuestr': lambda x: re.findall(r'(?!\t)\w+(?:\t$)', x)[0]
    },
    {   'key':'display fixation',
        'match':r'-?\d+\tDisplayFixation\t'
        },
]
feedback_map = [
    {   'key':'trial id',
        'match':r"^TRIALID",
        'valuestr': lambda x: float(re.findall(r'\d+',x)[0])
        },
    {   'key':'display fixation',
        'match':r'-?\d+\tDisplayFixation\t'
        },
    {   'key':'display target',
        'match':r"-?\d+\tDisplayTarget"
        },
    {   'key':'task type and trial end sentinel',
        'match':r"!V\tTRIAL_VAR\ttype",
        'valuestr': lambda x: re.findall(r'(?!\t)\w+(?:\t$)',x)[0] 
        },
    {   'key':'trial max count',
        'match':r"!V\tTRIAL_VAR\tTRIAL_MAX_COUNT",
        'valuestr': lambda x: re.findall(r'\d+',x)[0] 
        },
    {   'key':'task',
        'match':r"!V\tTRIAL_VAR\tTASK\t",
        'valuestr': lambda x: re.findall(r'(?!\t)\w+(?:\t$)',x)[0] 
        },
    {   'key':'block count',
        'match':r"!V\tTRIAL_VAR\tBLOCK_COUNT",
        'valuestr': lambda x: re.findall(r'\d+',x)[0] 
        },
    {   'key':'fixation delay',
        'match':r"!V\tTRIAL_VAR\tFixDelay\t",
        'valuestr': lambda x: int(re.findall(r'\d+',x)[0])
        },
    {   'key':'trial ok',
        'match':r"!V\tTRIAL_VAR\tTrial_OK\t",
        'valuestr': lambda x: int(re.findall(r'\d+',x)[0]) # 1 = ok, 0 = no
        },
    {   'key':'trial type',
        'match':r"!V\tTRIAL_VAR\tTRIAL_TYPE\t",
        'valuestr': lambda x: int(re.findall(r'\d',x)[0]) # 1 = right, 0 = left 
        },
    {   'key':'fixation error',
        'match':r"!V\tTRIAL_VAR\tfixation_error\t",
        'valuestr': lambda x: re.findall(r'(?!\t)\w+(?:\t$)',x) != 'False' 
        },
    {   'key':'target positions',
        'match':r"!V\tTRIAL_VAR\ttargetpositions\t",
        'valuestr': lambda x: re.findall(r'\[\(\d+,\t\d+\),\t\(\d+,\t\d+\)\](?:\t$)',x)[0] 
        }
]


     

class unittests:
    def __init__(self, test, *args):
        ''' runs tests for this file and functions therin.
            it has a number of testing facilities.
            that number is len(self.knowntests) which is a dict
            of testnames and corresponding test functions.

            the tests are pretty much call em and they go. 
            '''
        logger = self.getlogger(logging.INFO) # display this level
        self.knowntests = {
            'makefeedback_dat':self.makefeedback_dat, 
            'parse_edf_partial':self.parse_edf_partial,
            'test_str_writes':self.test_str_writes
        }
        if test not in self.knowntests:
            logger.critical('test %s not known. known tests are %s'%(\
                                test, self.knowntests.keys()))
            return
        self.knowntests[test](logger=logger, *args)

    def test_str_writes(self,logger=logging):
        
        f = open('test.bin','wb')
        f.truncate()
        tint = -9998
        tshort = 70
        tfloat = 3.1467
        tstr = 'fellowship of the ring'
        tlist1 = [4.5,7.2,6.8]
        tlist2 = [8.2,4.5,6]
        f.write(int2bin(tint))
        f.write(short2bin(tshort))
        f.write(float2bin(tfloat))
        f.write(str2bin(tstr,18))
        f.write(list2bin(tlist1,'float'))
        f.write(list2bin([tlist1,tlist2],'vector','float'))
        f.write(vector2bin([1,2,3,4,5],'short', logger=logger))
        f.write(vector2bin([],'int', logger=logger))
        f.close()
        f = open('test.bin', 'rb')
        rint = bin2int(f.read(4),log=logger)
        logger.info('%s:%s'%(tint, rint))

        rshort = bin2short(f.read(2),log=logger)
        logger.info('%s:%s'%(tshort, rshort))

        rfloat = bin2float(f.read(4), log = logger)
        logger.info('%s:%s'%(tfloat, rfloat))

        rstr = bin2str(f.read(18),log=logger)
        logger.info('%s:%s'%(tstr, rstr))

        rlist1 = read_list(f, bin2float, 4, 3,log=logger)
        logger.info('%s:%s'%(tlist1, rlist1))

        rvec = read_vec_list(f, bin2float, 4, 2,log=logger)

        logger.info(read_vec(f, bin2short, 2, log=logging))
        logger.info(read_vec(f,bin2int, 4, log=logging))
        f.close()


    def parse_edf_partial(self, edfpath, logger=logging):
        ''' this test reads an edf file, parses it and prints 
            stuff to the screen and a log file for review. '''
        mapping = {
            'feedback.txt':feedback_map,
            'temp.txt': temp_dis_map,
            'color.txt': color_map,
            'memory.txt':memory_map
        }

        f = open_edf(edfpath)
        trs = self.test_get_trial_sets(f, logger=logger)
        hdr = trs[0]    # first index is header
        trs = trs[1:]   # the remainder are actual trials
        ptrs = [] # used for saving the trials
        # parse the first one for testing
        
        parsed_trial = self.test_parse_trial_set(trs[0], logger=logger) 
        ptrs.append(parsed_trial)
        logger.info('%d lines were dropped from trial'%len(parsed_trial['dropped']))
        
        # print messages for easy planning
        logger.debug('Messages from parsed trial:')
        for m in parsed_trial['MSG']:
            logger.debug('   %s'%m)
        
        # parse the messsages
        messages = parse_messages(parsed_trial['MSG'], 
                                    mapping[edf_for_test],
                                    logger=logger)
        logger.info('%d messages parsed out of %d messages'%(len(messages), 
                   len(parsed_trial['MSG'])))
        for m in messages:
            logger.debug('  %s: %s  (%d)'%(m,messages[m]['value'],messages[m]['timestamp']))
        
        # display this first trial
        #self.plot_trial_trace(parsed_trial)


        # parse the remaining trials
        # for tr in trs[1:]: # skip the first trial
        #     parsed_trial = self.test_parse_trial_set(tr) # don't log these
        #     ptrs.append(parsed_trial)

        # self.overlay_trials(ptrs, logger=logger)

    def makefeedback_dat(self, edfpath, datname, logger=logging):
        ''' this will attempt to make a datfile from an edf file. 
            this assumes the edf is a switching with feedback task. 
            '''
        expdata = parse_edf(edfpath, feedback_map, logger=logger)
        trials = expdata['trials']
        # trials is a dict with keys [messages, samples, dropped] 
        antisaccade_trials = []
        for tr in trials:
            if 'task' in tr['messages']['info']:
                if 'Antisaccade' in tr['messages']['info']['task']['value'] :
                    antisaccade_trials.append(tr)

        logger.info('from %s trials, %s antisaccades have been selected'%(\
                    len(trials), len(antisaccade_trials)))

        for tr in antisaccade_trials:
            logger.info('>> %s'%tr['messages'])

        return
        f = open(datname,'wb')
        f.truncate()
        
        # directory
        f.write(str2bin('human_1', 12, logger=logger))
        f.write(int2bin(1, logger=logger)) # no of datasets
        f.write(int2bin(1, logger=logger)) # no of blocks in directory
        f.write(str2bin('',4, logger=logger)) # unused
        f.write(str2bin('04Mar-16',8, logger=logger)) # last updated
        f.write(str2bin('',32, logger=logger)) # unused
        
        # dataset header
        f.write(str2bin('SCH018',8, logger=logger))
        dset_size_loc = f.tell()
        f.write(int2bin(0, logger=logger)) # we have to overwrite this later. --------------------------------------
        f.write(str2bin('human_1', 12, logger=logger)) # dset id
        f.write(int2bin(2, logger=logger)) # block where it startes indexed from 1
        f.write(str2bin('COM',4, logger=logger)) # exp type
        
        go_to_block(f, 2) # go to start of dset
        start_dset_words  = f.tell()/4
        logger.info('dsets starts at byte %s. or word %s. i.e. block 2.'%(f.tell(), start_dset_words))
        
        
        f.write(str2bin('SCH018', 8, logger=logger)) # scheme name
        f.write(int2bin(0, logger=logger)) # length of dset in blocks have to override this later ---------------
        f.write(str2bin('human_1', 12, logger=logger)) # animal id
        f.write(str2bin('feedback',12, logger=logger)) # dataset id
        f.write(str2bin('04Mar-16', 8, logger=logger)) # date DDMMM-YY 
        f.write(int2bin(504000, logger=logger)) # time of exp in 10ths of second past midnight. 2pm  = 504000 10s of seconds
        f.write(str2bin('COM',4, logger=logger))
        # end of mandatory header
        
        # actual data
        f.write(int2bin(1, logger=logger)) # udata
        f.write(int2bin(1, logger=logger)) # andata
        f.write(int2bin(0, logger=logger)) # CMDATA
        f.write(int2bin(1, logger=logger)) # SP1ch
        f.write(int2bin(1, logger=logger)) # STRTCH
        f.write(int2bin(4, logger=logger)) # TERMCH
        f.write(int2bin(3, logger=logger)) # INWCH
        f.write(int2bin(5, logger=logger)) # REWCH
        f.write(int2bin(2, logger=logger)) # ENDCH
        f.write(float2bin(.002,logger=logger)) # tbase
        f.write(int2bin(3, logger=logger)) # stform
        f.write(int2bin(2, logger=logger)) # numptr
        stat_table_pointer_loc = f.tell()
        f.write(int2bin(0, logger=logger)) # loc of status table at the end ---------------------------------------
        f.write(int2bin(len(antisaccade_trials), logger=logger)) # NSEQ # no of trials for now lets just put one in to see if it can run at all
        f.write(int2bin(1296504643, logger=logger)) # random seed. idk...
        f.write(float2bin(0.140000000596, logger=logger)) # grace period for fixation (led1)
        f.write(float2bin(5.0, logger=logger)) # time to spot led1
        f.write(float2bin(0.140000000596, logger=logger)) # grace period for target (led2)
        f.write(float2bin(5.0, logger=logger)) # time to spot led1
        f.write(float2bin(0.5, logger=logger)) # spontim
        f.write(float2bin(1, logger=logger)) # ISDREW
        f.write(float2bin(1, logger=logger)) # ISNODREW
        f.write(float2bin(40, logger=logger)) # att low
        f.write(float2bin(43, logger=logger)) # att high
        f.write(float2bin(1, logger=logger)) # att inc
        f.write(int2bin(0, logger=logger)) # lapend
        f.write(str2bin('',8, logger=logger)) # SCAPEND, schema name for appended data. 
        
        # fixed parameters for respondent
        f.write(int2bin(0, logger=logger)) # length of fixed params in words # LFXPAR
        f.write(int2bin(0, logger=logger)) # no of vars. We're not including any # NFXPAR
        # normally we'd put some in here as a repeating group but not now. they may not be neccessary for analysis
        f.write(str2bin('_123456789_123456789_123456789_123456789_123456789_12345678\x00',60, logger=logger)) # comment
        # subtask paramters
        f.write(int2bin(1, logger=logger)) # I think all that matters is the stat table
        
        # subtpar 1
        f.write(int2bin(572, logger=logger)) # LSUBTAR (length of subtparvin 4 byte words ) 
        f.write(int2bin(74, logger=logger)) # NSUBTPAR (no vars in subtparv) 
        # subtparv1
        write_var(f, 'tasktyp', 1, 1, int2bin(29))
        write_var(f, 'paramset', 1, 1, int2bin(990620))
        write_var(f, 'wind1x', 2, 1, float2bin(4.0))
        write_var(f, 'wind1y', 2, 1, float2bin(4.0))
        write_var(f, 'wind2x', 2, 1, float2bin(10.0))
        write_var(f, 'wind2y', 2, 1, float2bin(10.0))
        write_var(f, 'spk1flg', 1, 1, int2bin(0))
        write_var(f, 'spk2flg', 1, 1, int2bin(0))
        write_var(f, 'windcod1', 1, 1, int2bin(1))
        write_var(f, 'windcod2', 1, 1, int2bin(1))
        write_var(f, 'l1sel', 1, 1, int2bin(9))
        write_var(f, 'l1int', 1, 1, int2bin(3))
        write_var(f, 'l1col', 1, 1, int2bin(3))
        write_var(f, 'l1strt', 2, 1, float2bin(0.750000059605))
        write_var(f, 'l1nreps', 1, 1, int2bin(1))
        write_var(f, 'wtscnt', 1, 1, int2bin(32))
        write_var(f, 'weights', 1, 32, list2bin([0,5,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],'int'))
        write_var(f, 'l2sel', 1, 32, list2bin([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],'int'))
        write_var(f, 'l2int', 1, 32, list2bin([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],'int'))
        write_var(f, 'l2col', 1, 32, list2bin([1,16711680,1,16711680,1,1,1,1,1,1,1,1,1,16711680,16711680,16711680,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],'int'))
        write_var(f, 'l2strt', 2, 32, list2bin([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float'))
        write_var(f, 's2sel', 1, 32, list2bin([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],'int'))
        write_var(f, 's2strt', 2, 32, list2bin([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float'))
        write_var(f, 'v1sel', 1, 32, list2bin([900,1050,1000,1000,900,1000,2000,1000,900,950,2000,1000,900,1050,1050,1000,900,1050,1000,950,900,1050,900,950,1000,1050,1050,1000,950,900,950,1050],'int'))
        write_var(f, 'v1int', 1, 32, list2bin([1,15,15,14,13,12,11,10,1,8,7,6,5,4,2,2,19,20,17,18,1,1,1,1,1,1,1,1,1,1,1,1],'int'))
        write_var(f, 'v1col', 1, 32, list2bin([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],'int'))
        write_var(f, 'v1strt', 2, 32, list2bin([2,2,2,2,2,2,2,2,2,2,2,2,2,2,898048,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2], 'float'))
        write_var(f, 'a1sel', 1, 32, list2bin([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],'int'))
        write_var(f, 'a1strt', 2, 32, list2bin([1,16,1,14,1,12,11,1,1,1,7,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 'float'))
        write_var(f, 'l1dur', 2, 32, list2bin([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'float'))
        write_var(f, 'l2width', 2, 1, float2bin(0.350000023842))
        write_var(f, 'l2period', 2, 1, float2bin(0.850000023842))
        write_var(f, 'l2nreps', 1, 1, int2bin(1))
        write_var(f, 'l2dur', 2, 1, float2bin(0.350000023842))
        write_var(f, 's2width', 2, 1, float2bin(0.800000011921))
        write_var(f, 's2period', 2, 1, float2bin(0.900000035763))
        write_var(f, 's2nreps', 1, 1, int2bin(1))
        write_var(f, 's2dur', 2, 1, float2bin(0.0))
        write_var(f, 'v1width', 2, 1, float2bin(0.350000023842))
        write_var(f, 'v1period', 2, 1, float2bin(0.850000023842))
        write_var(f, 'v1nreps', 1, 1, int2bin(1))
        write_var(f, 'v1dur', 2, 1, float2bin(0.350000023842))
        write_var(f, 'a1width', 2, 1, float2bin(0.300000011921))
        write_var(f, 'a1period', 2, 1, float2bin(0.555000007153))
        write_var(f, 'a1nreps', 1, 1, int2bin(1))
        write_var(f, 'a1dur', 2, 1, float2bin(0.0))
        write_var(f, 'stay2dur', 2, 1, float2bin(0.0500000007451))
        write_var(f, 's2type', 3,  3, str2bin('noi',3*4))
        write_var(f, 's2wdir', 3, 16, str2bin('',16*4))
        write_var(f, 's2wfil', 3, 16, str2bin('',16*4))
        write_var(f, 's2wres', 2, 1, float2bin(0.0))
        write_var(f, 's2wpts', 1, 1, int2bin(0))
        write_var(f, 's2clkwid', 2, 1, float2bin(0.0))
        write_var(f, 's2clkper', 2, 1, float2bin(0.0))
        write_var(f, 's2frq', 2, 1, float2bin(0.0))
        write_var(f, 's2trise', 2, 1, float2bin(0.0100000007078))
        write_var(f, 's2tfall', 2, 1, float2bin(0.0100000007078))
        write_var(f, 'a1type', 3,  3, str2bin('sin',3*4))
        write_var(f, 'a1wdir', 3, 16, str2bin('',16*4))
        write_var(f, 'a1wfil', 3, 16, str2bin('',16*4))
        write_var(f, 'a1wres', 2, 1, float2bin(0.0))
        write_var(f, 'a1wpts', 1, 1, int2bin(0))
        write_var(f, 'a1clkwid', 2, 1, float2bin(0.0))
        write_var(f, 'a1clkper', 2, 1, float2bin(0.0))
        write_var(f, 'a1frq', 2, 1, float2bin(1000.0))
        write_var(f, 'a1trise', 2, 1, float2bin(0.0100000007078))
        write_var(f, 'a1tfall', 2, 1, float2bin(0.0100000007078))
        write_var(f, 'gazechec', 1, 1, int2bin(1))
        write_var(f, 'okangle', 2, 1, float2bin(90.0))
        write_var(f, 'rewvary', 1, 1, int2bin(0))
        write_var(f, 'rw1ontim', 2, 1, float2bin(250.0))
        write_var(f, 'rw2ontim', 2, 1, float2bin(0.0))
        write_var(f, 'rw3ontim', 2, 1, float2bin(0.0))
        write_var(f, 'rw4ontim', 2, 1, float2bin(0.0))



        # END OF SUBPARV FOR ANTISACCADE i.e. task 29
        f.write(float2bin(.000305175, logger=logger)) # AVOLC
        f.write(int2bin(1, logger=logger)) # AVCC voltage conversion code.... 
        f.write(int2bin(16, logger=logger)) # ANBITS
        
        # A/D channels
        f.write(int2bin(2, logger=logger)) # NACH
        #a/d cahnnel 0 for x
        f.write(int2bin(0, logger=logger)) # Achan number
        f.write(float2bin(500, logger=logger)) # SRATE. samples/sec
        f.write(float2bin(0, logger=logger)) # ASAMPT
        f.write(int2bin(0, logger=logger)) # nsamp..... always 0?
        #a/d cahnnel 1 for y
        f.write(int2bin(1, logger=logger)) # Achan number
        f.write(float2bin(500, logger=logger)) # SRATE. samples/sec
        f.write(float2bin(0, logger=logger)) # ASAMPT
        f.write(int2bin(0, logger=logger)) # nsamp..... always 0?
        
        f.write(int2bin(1, logger=logger)) # coil code. means there are coils. I think
        f.write(int2bin(2, logger=logger)) # ncoilcofs
        f.write(int2bin(1, logger=logger)) # ncoil
        # coil 1
        f.write(str2bin('RITEYE',8, logger=logger)) # RITEYE, LEFTEYE determined from 
        f.write(int2bin(0, logger=logger)) # ADCHX a/d channel for x
        f.write(int2bin(1, logger=logger)) # ADCHY a/d channel for y
        # coil 1 cofs 1
        f.write(float2bin(1, logger=logger)) # COFx
        f.write(float2bin(1, logger=logger)) # COFy
        # coil 1 cofs 2
        f.write(float2bin(1, logger=logger)) # COFx
        f.write(float2bin(1, logger=logger)) # COFy
        
        f.write(float2bin(0, logger=logger)) # avolccm conversion factor. no ccm 0
        f.write(float2bin(0, logger=logger)) # avccm. conversion code. no ccm 0
        f.write(int2bin(0, logger=logger)) # anbitscm. no cm 0
        f.write(int2bin(0, logger=logger)) # achannm. no an chans devoted to cm. 0
        f.write(int2bin(0, logger=logger)) # numCm / no cm 0
        
        f.write(int2bin(0, logger=logger)) # numleds
        f.write(int2bin(0, logger=logger)) # numspk
        f.write(int2bin(8, logger=logger)) # ldummy
        f.write(str2bin('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',8*4, logger=logger)) # dummy
        
        # data
        uet_locs_word = []
        andata_locs_word = []
        for i,tr in enumerate(antisaccade_trials): # same as NSEQ = len(antisaccade_trials)
            logger.info('trial %s out of %s (%s)'%(i+1,len(antisaccade_trials), f.tell()))
            uet_loc_word = f.tell()/4 + 1
            logger.info('for trial %s uetdata is at word %s'%(i+1, uet_loc_word))
            uet_locs_word.append(uet_loc_word)
            uet_arr = [uet2bin(ch, time) for ch,time in tr['uets']]
            tsdatastr = vector2bin(uet_arr,'bin',logger=logger)
            logger.info('length of vector str should be 24. is %s'%len(tsdatastr))
            logger.info('# about to write tsdata. loc = %s'%f.tell())
            f.write(tsdatastr) # TSdata. we have no spikes
            
            andata_loc_word = f.tell()/4 + 1
            logger.info('for trial %s andata is at word %s'%(i+1, andata_loc_word))
            andata_locs_word.append(andata_loc_word)
            xdata = [s['xpos'] for s in tr['samples']]
            ydata = [s['ypos'] for s in tr['samples']]
            andata = [xdata,ydata]
            andatastr = list2bin(andata,'vector','short', logger=logger) 
            logger.info('andata binstr length is %s. it should be %s =- 4 bytes for padding'%(len(andatastr), 2*(2*len(xdata) + 4)))
            f.write(andatastr)
            # we would have cmdata. but..... no cm. no care
            f.write(int2bin(0, logger=logger)) # NDERVS

        # stat table
        loc_stat_table_words = f.tell()/4 + 1 # move past first nvstat, idexed oddly
        for i,tr in enumerate(antisaccade_trials):
            logger.info('\n\nstattable entry %s'%(i+1))
            logger.info('NVSTAT : 4')
            f.write(int2bin(36, logger=logger)) # NVSTAT no entries in stattable
            write_var(f, 'tasktyp', 1, 1, int2bin(29))
            write_var(f, 'paramset', 1, 1, int2bin(990620))
            write_var(f, 'attset',2, 1, float2bin(40.0))
            write_var(f, 'wind1x',2, 1, float2bin(5.0))
            write_var(f, 'wind1y',2, 1, float2bin(5.0))
            write_var(f, 'wind2x',2, 1, float2bin(5.0))
            write_var(f, 'wind2y',2, 1, float2bin(5.0))
            write_var(f, 'rewcur', 1, 1, int2bin(1))
            write_var(f, 'led1ch', 1, 1, int2bin(9))
            write_var(f, 'spk1ch', 1, 1, int2bin(10))
            write_var(f, 'led2ch', 1, 1, int2bin(7))
            write_var(f, 'spk2ch', 1, 1, int2bin(11))
            write_var(f, 'l2sel', 1, 1, int2bin(14))
            write_var(f, 'l2int', 1, 1, int2bin(3))
            write_var(f, 'l2col', 1, 1, int2bin(16711680))
            write_var(f, 'l2strt',2, 1, float2bin(0.0))
            write_var(f, 's2sel', 1, 1, int2bin(14))
            write_var(f, 's2strt',2, 1, float2bin(0.0))
            write_var(f, 'l1dur',2, 1, float2bin(0.900000035763))
            write_var(f, 's1type',3,3,str2bin('noi',4*3)         )
            write_var(f, 's1wdir',3,16,str2bin('',4*16) )
            write_var(f, 's1wfil',3,16,str2bin('',4*16) )
            write_var(f, 's2wres',2, 1, float2bin(0.0))
            write_var(f, 's2wpts', 1, 1, int2bin(0))
            write_var(f, 's2clkwid',2, 1, float2bin(0.0))
            write_var(f, 's2clkper',2, 1, float2bin(0.0))
            write_var(f, 's2frq',2, 1, float2bin(0.0))
            write_var(f, 's2trise',2, 1, float2bin(0.0100000007078))
            write_var(f, 's2tfall',2, 1, float2bin(0.0100000007078))
            write_var(f, 'success', 1, 1, int2bin(2))
            write_var(f, 'timemsec', 1, 1, int2bin(0))
            write_var(f, 'npcmd', 1, 1, int2bin(0))
            write_var(f, 'swichmrk', 1, 1, int2bin(0))
            write_var(f, 'extratim',2, 1, float2bin(0.0))
            write_var(f, 'l1imgfil',3,16,str2bin('',4*16))
            write_var(f, 'l2imgfil',3,16,str2bin('',4*16))

            # write addr pointers
            logger.info('UET loc: %s'%(uet_locs_word[i] - start_dset_words))
            f.write(int2bin(uet_locs_word[i] - start_dset_words,logger=logger))
            logger.info('ANDATA loc: %s'%(andata_locs_word[i] - start_dset_words))
            f.write(int2bin(andata_locs_word[i] - start_dset_words, logger=logger))


        logger.critical('End of file is at byte %s'%f.tell())
        # skip back to STAT
        loc_stat_table_offset = loc_stat_table_words - start_dset_words
        f.seek(stat_table_pointer_loc)
        f.write(int2bin(loc_stat_table_offset,logger=logger))
        logger.info('loc of stat table in offset is %s. check that using parsedatfile.py'%loc_stat_table_offset)
        logger.info('cloding file. loc: %s'%f.tell())
        f.close()

        logger.info('Done building datafile!')

    def getlogger(self, consoleLevel,
        filename = 'log.md',
        filemode='w',
        fileLevel = logging.DEBUG, 
        formatstr = "%(levelname)s:%(funcName)s>>>  %(message)s"):
        ''' sets up the logger 
            consoleLevel is the level that will be displayed  on console '''
        log_format = logging.Formatter(formatstr)
        logger = logging.getLogger("unittests") # grabs a logger from logging
        logger.setLevel(logging.DEBUG) # sets base things to log

        fh = logging.FileHandler(filename, mode=filemode) # get thing that handles files
        fh.setLevel(fileLevel) # make sure whe're printing only what we want
        fh.setFormatter(log_format) # in the correct format
        logger.addHandler(fh) # add the file handler

        sh = logging.StreamHandler() # get thing that handles the console
        sh.setLevel(consoleLevel) # make sure that we're only printing what we want 
        sh.setFormatter(log_format) # in the correct format
        logger.addHandler(sh) # add the console handler 
        return logger # return 

    def test_get_trial_sets(self, f, logger=logging):
        ''' tests get_trials function '''
        trial_sets = get_trial_sets(f)
        logger.info('Found %d trials in the file'%(len(trial_sets) -1))
        if len(trial_sets) > 1:
            logger.info('First line of trial 1 looks like: %s'%trial_sets[1][0])
        return trial_sets

    def test_parse_trial_set(self, tr_set, logger=logging):
        ''' tests parse_trial_set function ''' 
        logger.info('-----')
        logger.info('----------')
        logger.info('-----------------------')
        logger.info( 'testing parse_trial_set function')
        events = parse_trial_set(tr_set, org='multi', logger=logger) # just use default flags
        logger.info('%d trials passed in for this set'%len(tr_set))
        logger.info('%d types were found'%len(events.keys()))
        samples, messages = [],[]
        if 'sample' in events: samples = events['sample']
        if 'MSG' in events: messages =events['MSG'] 
        logger.info('%d trials parsed during operation'%(len(events['MSG']) + len(events['sample'])))
        logger.info('%d messages, and %d samples'%(len(events['MSG']), len(events['sample'])))

        logger.info('')
        logger.info('First message looks like so: ')
        m = messages[0]
        for i in m:
            logger.info('   %s : %s'%(i,str(m[i])))
        logger.info('First sample looks like so:')
        s = samples[0]
        for i in s:
            logger.info('   %s : %s'%(i,str(s[i])))

        return events

    def plot_trial_trace(self, parsed_trial, logger=logging, ax=None):
        ''' this will plot a sinlgle trial from the memory guided 
            saccade task. 

            parsed trial assumes a parsed trial in the multi format from 
            parse_trial_set i.e. a dict with samples and messages and 
                potentially others

            you can either pass an existing axis or create a new one. 
            if you create a newone this function will plot it and halt.
            elseit will wait for you to call ax.show()

            '''
        # this will plot time along the x axis
        # and the x/y traces of the eye along the y axis
        assert type(parsed_trial) is dict, \
                'Type of parsed trial is %s'%type(parsed_trial)
        assert 'sample' in parsed_trial, 'key "sample" is not in parsed_trial'
        samples = parsed_trial['sample']
        t = []
        x = []
        y = []
        try:
            t = [i['timestamp'] for i in samples]
            x = [i['xpos'] for i in samples]
            y = [i['ypos'] for i in samples]
        except KeyError,e:
            logger.error('Cannot plot. sample is not as expected. Error %s'%e)
        
        for i in t: 
            assert type(i) is float, 'Type of %s is %s'%(i, type(i))
        for i in x: 
            assert type(i) is float, 'Type of %s is %s'%(i, type(i))
        for i in y: 
            assert type(i) is float, 'Type of %s is %s'%(i, type(i))

        show = False
        if ax is None: 
            show = True
            ax = plt.subplot()
        ax.plot(t,x,'g', label='x')
        ax.plot(t,y,'b', label='y')

        if show: 
            legend = ax.legend(shadow=True)
            plt.show()
        return ax

    def overlay_trials (self, parsed_trial_sets, logger=logging):
        ''' this plots many trial sets. taken as a list of parsed_trial_sets
            then plots them all
            '''
        ax = plt.subplot() # create the axes to plot all these guys
        for i,trial in enumerate(parsed_trial_sets):
            logger.debug('adding trace for trial # %d'%i)
            ax = self.plot_trial_trace(trial, logger=logger, ax=ax)
        plt.show()


# unittests('test_str_writes')
unittests('makefeedback_dat', ['feedback.txt'],'RL_hu_test.dat')

