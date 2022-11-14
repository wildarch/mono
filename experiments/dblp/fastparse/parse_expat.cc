#include <stdio.h>
#include <string.h>
#include <expat.h>

static void XMLCALL
startElement(void *userData, const XML_Char *name, const XML_Char **atts)
{
    int *const depthPtr = (int *)userData;
    (void)atts;

    if (strcmp(name, "poep") == 0)
    {
        printf("Poep at depth %d\n", *depthPtr);
    }

    *depthPtr += 1;
}

static void XMLCALL
endElement(void *userData, const XML_Char *name)
{
    int *const depthPtr = (int *)userData;
    (void)name;

    *depthPtr -= 1;
}

int main(void)
{
    XML_Parser parser = XML_ParserCreate(NULL);
    int done;
    int depth = 0;

    if (!parser)
    {
        fprintf(stderr, "Couldn't allocate memory for parser\n");
        return 1;
    }

    XML_SetUserData(parser, &depth);
    XML_SetElementHandler(parser, startElement, endElement);

    do
    {
        void *const buf = XML_GetBuffer(parser, BUFSIZ);
        if (!buf)
        {
            fprintf(stderr, "Couldn't allocate memory for buffer\n");
            XML_ParserFree(parser);
            return 1;
        }

        const size_t len = fread(buf, 1, BUFSIZ, stdin);

        if (ferror(stdin))
        {
            fprintf(stderr, "Read error\n");
            XML_ParserFree(parser);
            return 1;
        }

        done = feof(stdin);

        if (XML_ParseBuffer(parser, (int)len, done) == XML_STATUS_ERROR)
        {
            fprintf(stderr,
                    "Parse error at line %lu:\n%s\n",
                    XML_GetCurrentLineNumber(parser),
                    XML_ErrorString(XML_GetErrorCode(parser)));
            XML_ParserFree(parser);
            return 1;
        }
    } while (!done);

    XML_ParserFree(parser);
    return 0;
}