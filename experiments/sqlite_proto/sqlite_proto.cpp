#include "sqlite3ext.h"
#include <google/protobuf/message.h>
#include <google/protobuf/unknown_field_set.h>
#include <google/protobuf/empty.pb.h>
// TODO: remove
#include <iostream>

SQLITE_EXTENSION_INIT1

static void addValue(google::protobuf::UnknownFieldSet *fieldSet, int number, sqlite3_value *value)
{
    std::string strVal;
    switch (sqlite3_value_type(value))
    {
    case SQLITE_INTEGER:
        fieldSet->AddVarint(number, sqlite3_value_int64(value));
        break;
    case SQLITE_FLOAT:
        fieldSet->AddFixed64(number, (uint64_t)sqlite3_value_double(value));
        break;
    case SQLITE_TEXT:
        strVal = std::string((const char *)sqlite3_value_text(value), sqlite3_value_bytes(value));
        fieldSet->AddLengthDelimited(number, strVal);
        break;
    case SQLITE_BLOB:
        strVal = std::string((const char *)sqlite3_value_blob(value), sqlite3_value_bytes(value));
        fieldSet->AddLengthDelimited(number, strVal);
        break;
    case SQLITE_NULL:
        // Skip
        break;
    }
}

static void extractFunc(sqlite3_context *context, int argc,
                        sqlite3_value **argv)
{
    assert(argc == 2);
    int number = sqlite3_value_int(argv[0]);
    google::protobuf::Empty proto;
    proto.ParseFromArray(sqlite3_value_blob(argv[1]), sqlite3_value_bytes(argv[1]));
    auto *reflect = proto.GetReflection();
    auto *fieldSet = reflect->MutableUnknownFields(&proto);
    std::cout << "fieldsSet len: " << fieldSet->field_count() << std::endl;
    const google::protobuf::UnknownField *field = nullptr;
    for (int i = 0; i < fieldSet->field_count(); i++)
    {
        auto &f = fieldSet->field(i);
        std::cout << "Field number: " << f.number() << std::endl;
        if (f.number() == number)
        {
            field = &f;
            break;
        }
    }
    if (field == nullptr)
    {
        sqlite3_result_error(context, "field not found", -1);
        return;
    }
    void *resultBuf = nullptr;
    switch (field->type())
    {
    case google::protobuf::UnknownField::Type::TYPE_VARINT:
        sqlite3_result_int64(context, field->varint());
        break;
    case google::protobuf::UnknownField::Type::TYPE_FIXED32:
        sqlite3_result_int(context, field->fixed32());
        break;
    case google::protobuf::UnknownField::Type::TYPE_FIXED64:
        sqlite3_result_int64(context, field->fixed64());
        break;
    case google::protobuf::UnknownField::Type::TYPE_LENGTH_DELIMITED:
        resultBuf = sqlite3_malloc64(field->GetLengthDelimitedSize());
        mempcpy(resultBuf, field->length_delimited().data(), field->GetLengthDelimitedSize());
        sqlite3_result_text(context, (char *)resultBuf, field->GetLengthDelimitedSize(), sqlite3_free);
        break;
    case google::protobuf::UnknownField::Type::TYPE_GROUP:
        // TODO: Handle
        break;
    }
}

static void buildFunc(sqlite3_context *context, int argc,
                      sqlite3_value **argv)
{
    if (argc % 2 != 0)
    {
        return sqlite3_result_error(context, "proto requires pairs of tag and value (an even number of inputs)", -1);
    }
    /*
    sqlite3_int64 bufSize = 64;
    void *buf = nullptr;
    bool hadError;
    do
    {
        hadError = false;
        buf = sqlite3_realloc64(buf, bufSize);
        google::protobuf::io::ArrayOutputStream aos(buf, bufSize);
        google::protobuf::io::CodedOutputStream cos(&aos);

        for (int i = 0; i < argc; i += 2)
        {
            // Skip null values
            if (sqlite3_value_type(argv[i + 1]))
            {
                continue;
            }
            uint32_t tag = sqlite3_value_int(argv[i]);
            cos.WriteTag(tag);
            writeValue(cos, argv[i + 1]);

            if (cos.HadError())
            {
                // The buffer was too small, restart with a larger one.
                bufSize *= bufSize;
                hadError = true;
                break;
            }
        }
    } while (hadError);
    sqlite3_free(buf);
    */
    google::protobuf::Empty proto;
    auto *fieldSet = proto.GetReflection()->MutableUnknownFields(&proto);
    for (int i = 0; i < argc; i += 2)
    {
        addValue(fieldSet, sqlite3_value_int(argv[i]), argv[i + 1]);
    }
    auto serialized = proto.SerializePartialAsString();
    void *serializedBuf = sqlite3_malloc64(serialized.length());
    memcpy(serializedBuf, serialized.data(), serialized.length());
    sqlite3_result_blob64(context, serializedBuf, serialized.length(), sqlite3_free);
}

extern "C"
{
    int sqlite3_sqliteproto_init(sqlite3 *db, char **pzErrMsg,
                                 const sqlite3_api_routines *pApi)
    {
        int rc = SQLITE_OK;
        SQLITE_EXTENSION_INIT2(pApi);
        (void)pzErrMsg; /* Unused parameter */

        rc = sqlite3_create_function(db, "proto_extract", 2,
                                     SQLITE_UTF8 | SQLITE_INNOCUOUS |
                                         SQLITE_DETERMINISTIC,
                                     0, extractFunc, 0, 0);
        if (rc != SQLITE_OK)
        {
            return rc;
        }

        rc = sqlite3_create_function(db, "proto", -1,
                                     SQLITE_UTF8 | SQLITE_INNOCUOUS |
                                         SQLITE_DETERMINISTIC,
                                     0, buildFunc, 0, 0);

        return rc;
    }
}