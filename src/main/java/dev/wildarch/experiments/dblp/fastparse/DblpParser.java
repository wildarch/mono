package dev.wildarch.experiments.dblp.fastparse;

import com.ctc.wstx.api.WstxInputProperties;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLResolver;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.events.XMLEvent;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;

public class DblpParser {
    public static void main(String[] args) throws XMLStreamException {
        String dtdPath = args[0];
        XMLInputFactory xmlInputFactory = XMLInputFactory.newInstance();
        xmlInputFactory.setProperty(WstxInputProperties.P_MAX_ENTITY_COUNT, 10000000);
        xmlInputFactory.setXMLResolver((publicID, systemID, baseURI, namespace) -> {
            try {
                return Files.newInputStream(Paths.get(dtdPath));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
        XMLEventReader reader = xmlInputFactory.createXMLEventReader(System.in);
        while (reader.hasNext()) {
            XMLEvent nextEvent = reader.nextEvent();
        }
    }
    /* Basic version without GZIP
    public static void main(String[] args) throws ParserConfigurationException, SAXException, IOException {
        // we need to raise entityExpansionLimit because the dblp.xml has millions of entities
        System.setProperty("entityExpansionLimit", "10000000");

        System.out.println("Hello, world!");
        String dtdPath = args[0];
        InputStream dblpXmlStream = System.in;

        SAXParserFactory parserFactory = SAXParserFactory.newInstance();
        SAXParser parser = parserFactory.newSAXParser();
        parser.parse(dblpXmlStream, new DefaultHandler() {
            @Override
            public InputSource resolveEntity(@SuppressWarnings("unused") String publicId,
                                             @SuppressWarnings("unused") String systemId)
                    throws IOException {

                return new InputSource(Files.newInputStream(Paths.get(dtdPath)));
            }
        });
    }
     */

    /* Version with GZIP built-in
    public static void main(String[] args) {
        // we need to raise entityExpansionLimit because the dblp.xml has millions of entities
        System.setProperty("entityExpansionLimit", "10000000");

        System.out.println("Hello, world!");
        String dblpGzPath = args[0];
        String dtdPath = args[1];
        try (InputStream dblpXmlStream = new GZIPInputStream(Files.newInputStream(Paths.get(dblpGzPath)))) {
            SAXParserFactory parserFactory = SAXParserFactory.newInstance();
            SAXParser parser = parserFactory.newSAXParser();
            parser.parse(dblpXmlStream, new DefaultHandler() {
                @Override
                public InputSource resolveEntity(@SuppressWarnings("unused") String publicId,
                                                 @SuppressWarnings("unused") String systemId)
                        throws IOException {

                    return new InputSource(Files.newInputStream(Paths.get(dtdPath)));
                }
            });
        } catch (IOException e) {
            System.err.println("Failed to open/read dblp file: " + e);
            System.exit(1);
        } catch (ParserConfigurationException | SAXException e) {
            System.err.println("SAX parse error: " + e);
            System.exit(1);
        }
    }
     */
}
