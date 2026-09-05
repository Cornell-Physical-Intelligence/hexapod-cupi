# First live overlap attempt: diagnostic API failure

The filtered case on frozen production source d863663 started on the GPU but
failed at its first contact-count comparison: `NotImplementedError:
"compare_cuda" not implemented for UInt32`. No complete contact trace exists,
no control expectation was met, and the negative case was not launched.

The supervisor rejected the incomplete final report despite Kit exiting zero,
captured the native log and removed only its owned immutable container. Source
and external fixture remained unchanged. This is a diagnostic implementation
failure; it proves neither collision isolation nor a physical model failure.
The source archive remains on Spark; the compact evidence includes the exact
external fixture archive and the input hashes. The next separately staged
fixture converts native unsigned counts losslessly to int64 before comparisons
and records their native dtype. All admission flags remain false.
