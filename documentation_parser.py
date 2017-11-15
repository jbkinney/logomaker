import re
import inspect
import validate
import warnings

class ParameterDoc:
    def __init__(self, entry, sec_name, num_in_sec, param_num):
        self.section = sec_name
        self.num_in_section = num_in_sec
        self.param_num = param_num
        pattern = re.compile(
            "(?P<name>\w+)[ ]+\((?P<in_type>[\S \n]+)\)[ ]*:(?P<description>(?:[ ]*\S+[\S ]*\n)+)")

        m = re.match(pattern, entry)
        if m is not None:
            self.name = m.group('name')
            self.in_type = m.group('in_type')
            self.description = m.group('description').strip()
            self.set_default(None)
        else:
            self.name = None
            self.repr = 'None'

    def set_default(self, default):
        """ Set default value for parameter. """
        self.default = default
        self.repr = '%s (%s): %s\n    Default %s.\n' % \
                    (self.name, self.in_type, self.description,
                     repr(self.default))

    def get_argparse_doc(self):
        """ Get documentation for use in argparse. """
        doc = self.description
        doc = re.sub('%', '%%', doc)
        doc = re.sub('\n', '', doc)
        doc += ' Default %s.' % repr(self.default)
        return doc

    def __repr__(self):
        """ Get a string representation of this object. """
        return self.repr


def parse_documentation_file(file_name):
    # Open documentation file
    doc_contents = open(file_name, 'r').read()

    # Remove strange returns
    doc_contents = re.sub('\n\r', '\n', doc_contents)
    doc_contents = re.sub('\r\n', '\n', doc_contents)
    doc_contents = re.sub('\r', '\n', doc_contents)

    # Replace tabs with spaces
    doc_contents = re.sub('\t', '    ', doc_contents)

    # Create list to hold contents of documentation
    doc_dict = {}

    # Parse doc_contents into sections
    sec_pattern = re.compile(
        '(?<=###)(?P<heading>[^#\n]+)\n(?P<section>[\S \n]+?)(?=###)')
    sec_matches = re.finditer(sec_pattern, doc_contents)
    param_num = 0
    for i, sec_match in enumerate(sec_matches):
        sec_name = sec_match.group('heading').strip()
        sec_contents = sec_match.group('section')

        # Parse parameter entries
        param_pattern = re.compile(
            "\n[ ]*(?P<entry>\w+(?:[ ]*\S+[\S ]*\n)+)")

        param_matches = re.finditer(param_pattern, sec_contents)
        for j, param_match in enumerate(param_matches):
            entry = param_match.group('entry')
            doc = ParameterDoc(entry=entry,
                               sec_name=sec_name,
                               num_in_sec=j,
                               param_num=param_num)
            if doc.name is not None:
                doc_dict[doc.name] = doc
            else:
                warnings.warn('Could not parse documentation entry %s' % entry)
            doc.param_num += 1
            param_num += 1

    # # Parse entries
    # pattern = re.compile("\n[ ]*(?P<entry>\w+(?:[ ]*\S+[\S ]*\n)+)")
    # matches = re.finditer(pattern, doc_contents)
    #
    # # Create ordered list of ParameterDoc objects
    # doc_list = []
    # for i, m in enumerate(matches):
    #     entry = m.group('entry')
    #     doc = ParameterDoc(entry)
    #     if doc.name is not None:
    #         doc_list.append(doc)
    #     else:
    #         warnings.warn('Could not parse documentation entry %s' % entry)

    # Return dictionary to user
    return doc_dict


def document_function(func, doc_file):
    # Parse make_logo documentation file
    doc_dict = parse_documentation_file(doc_file)

    # Add default value to each parameter
    docstr = func.__doc__
    names, vargs, kwargs, default_values = inspect.getargspec(func)
    for i, name in enumerate(names):
        if name in doc_dict:

            # Get entry object based on parameter name
            entry = doc_dict[name]

            # Set default value in entry
            entry.set_default(default_values[i])

            # Get string version with additional whitespace on each line
            entry_str = repr(entry)
            entry_str = '\n' + '\n'.join([' '*8 + line
                                        for line in entry_str.split('\n')])

            # Recor string version of entry
            docstr += entry_str
        else:
            warnings.warn('Parameter %s not documented.' % name)
            docstr += '\n' + ' '*8 + '%s: NOT DOCUMENTED.\n' % name

    # Set docstring
    func.__doc__ = docstr



