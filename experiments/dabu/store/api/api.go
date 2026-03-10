package api

type StoreId [32]byte

type PutFileRequest struct {
	Data []byte
}

type PutResponse struct {
	Id StoreId
}

type PutDirRequest struct {
	Children []PutDirRequestChild
}

type PutDirRequestChild struct {
	Name string
	Id   StoreId
}

type GetFileRequest struct {
	Root StoreId
	Path string
}

type GetFileResponse struct {
	Data []byte
}
