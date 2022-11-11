#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include <libxml/parserInternals.h>

void start_element_callback(void *user_data, const xmlChar *name, const xmlChar **attrs)
{
    printf("Beginning of element : %s \n", name);
    while (NULL != attrs && NULL != attrs[0])
    {
        printf("attribute: %s=%s\n", attrs[0], attrs[1]);
        attrs = &attrs[2];
    }
}

void error_callback(void *user_data, const char *msg, va_list vargs)
{
    fprintf(stderr, msg, vargs);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Missing argument: path to XML document\n");
        return EXIT_FAILURE;
    }
    const char *xml_path = argv[1];

    // Initialize all fields to zero
    xmlSAXHandler sh = {0};

    // register callback
    sh.startElement = start_element_callback;
    sh.error = (errorSAXFunc)error_callback;

    xmlParserCtxtPtr ctxt;

    // create the context
    if ((ctxt = xmlCreateFileParserCtxt(xml_path)) == NULL)
    {
        fprintf(stderr, "Erreur lors de la création du contexte\n");
        return EXIT_FAILURE;
    }
    if (xmlCtxtUseOptions(ctxt, XML_PARSE_DTDLOAD | XML_PARSE_DTDVALID) != XML_ERR_NONE)
    {
        fprintf(stderr, "Failed to enable DTD loading.\n");
        return EXIT_FAILURE;
    }
    // register sax handler with the context
    ctxt->sax = &sh;

    // parse the doc
    xmlParseDocument(ctxt);
    // well-formed document?
    if (ctxt->wellFormed)
    {
        printf("XML Document is well formed\n");
    }
    else
    {
        fprintf(stderr, "XML Document isn't well formed\n");
        // xmlFreeParserCtxt(ctxt);
        return EXIT_FAILURE;
    }

    // free the memory
    xmlFreeParserCtxt(ctxt);

    printf("Done!\n");

    return EXIT_SUCCESS;
}