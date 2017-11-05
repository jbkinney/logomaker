import re
import inspect
import validate
import warnings

class ParameterDoc:
    def __init__(self, entry):
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

    # Parse entries
    pattern = re.compile("\n[ ]*(?P<entry>\w+(?:[ ]*\S+[\S ]*\n)+)")
    matches = re.finditer(pattern, doc_contents)

    # Create dictionary of ParameterDoc objects
    doc_dict = {}
    for i, m in enumerate(matches):
        entry = m.group('entry')
        doc = ParameterDoc(entry)
        if doc.name is not None:
            doc_dict[doc.name] = doc
        else:
            warnings.warn('Could not parse documentation entry %s' % entry)

    # Return dictionary to user
    return doc_dict


def document_function(func, doc_file):
    # Parse make_logo documentation file
    doc_dict = parse_documentation_file(doc_file)

    # Add default value to each parameter
    docstr = ""
    names, vargs, kwargs, default_values = inspect.getargspec(func)
    for i, name in enumerate(names):
        if name in doc_dict:

            # Get entry object based on parameter name
            entry = doc_dict[name]

            # Set default value in entry
            entry.set_default(default_values[i])

            # Recor string version of entry
            docstr += '\n%s' % repr(entry)
        else:
            warnings.warn('Parameter %s not documented.' % name)
            docstr += '\n%s: NOT DOCUMENTED.\n' % name

    # Set docstring
    func.__doc__ = docstr



