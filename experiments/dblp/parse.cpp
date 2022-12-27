#include <fstream>
#include <iostream>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <xmlwrapp/xmlwrapp.h>
#include <iostream>
#include <exception>

/*
 * Here we create a class that will receive the parsing events.
 */
class myparser : public xml::event_parser
{
public:
    myparser()
    {
        std::cout << "myparser constructor\n";
    }

    ~myparser() override
    {
        std::cout << "myparser destructor\n";
    }

private:
    bool start_element(const std::string &name, const attrs_type &) override
    {
        std::cout << "begin tag '" << name << "'\n";
        return true;
    }

    bool end_element(const std::string &name) override
    {
        std::cout << "end tag '" << name << "'\n";
        return true;
    }

    bool text(const std::string &) override
    {
        return true;
    }

    bool warning(const std::string &message) override
    {
        std::cout << "warning: " << message << std::endl;
        return true;
    }
};

int main(int argc, char **argv)
{
    std::ifstream file(argv[1], std::ios_base::in | std::ios_base::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
    in.push(boost::iostreams::gzip_decompressor());
    in.push(file);
    std::istream instream(&in);

    myparser parser;
    if (!parser.parse_stream(instream))
    {
        std::cerr << "error parsing XML: " << parser.get_error_message() << std::endl;
    }
}