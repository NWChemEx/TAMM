#define MPI_THREAD_STRING(level)  \
( level==MPI_THREAD_SERIALIZED ? "THREAD_SERIALIZED" : \
        ( level==MPI_THREAD_MULTIPLE ? "THREAD_MULTIPLE" : \
                ( level==MPI_THREAD_FUNNELED ? "THREAD_FUNNELED" : \
                        ( level==MPI_THREAD_SINGLE ? "THREAD_SINGLE" : "ERROR!" ) ) ) )

