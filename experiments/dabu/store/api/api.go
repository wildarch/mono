package api

type StoreId [32]byte

type DirListing struct {
	Name string
	Id   StoreId
}

type PutFileRequest struct {
	Data         []byte
	PathForDebug string
}

type PutResponse struct {
	Id    StoreId
	IsNew bool
}

type PutDirRequest struct {
	Children []DirListing
}

type GetFileRequest struct {
	Root StoreId
	Path string
}

type GetFileResponse struct {
	Data []byte
}

type GetRequest struct {
	Id StoreId
}

type GetResponse struct {
	Data     []byte
	Children []DirListing
}
