# flipprep arm table (sourced). arm_setup <arm>: sets BIN, ENVS, ARGS.
#   A  = origin/main defaults (binary intarsia-main), no env, no flags
#   F  = the flip (binary intarsia-flip), no env, no flags: the recipe is the whole configuration
#   R  = the flip binary with --spec off (the ids reference)
#   F0 = the flip with --mtp-draft-vocab 0 (acceptance reference: the full head)
#   AO = main with the mtp2s opt-in set and 131072 (what the flip should equal)
#   FR = the flip with the revert switches (INTARSIA_BATCHED_MTP=1 INTARSIA_BATCHED_VERIFY=0 INTARSIA_VERIFY_TOPK=0 --mtp-draft-vocab 0): main's defaults
B='~/intarsia-flipprep/bin'
arm_setup() {
  ENVS="X=1"; ARGS=""
  case ${1%%.*} in
    A)  BIN=$B/intarsia-main ;;
    F)  BIN=$B/intarsia-flip ;;
    R)  BIN=$B/intarsia-flip; ARGS="--spec off" ;;
    F0) BIN=$B/intarsia-flip; ARGS="--mtp-draft-vocab 0" ;;
    AO) BIN=$B/intarsia-main; ENVS="INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1 INTARSIA_VERIFY_TOPK=1"; ARGS="--spec mtp --spec-k 2 --mtp-draft-vocab 131072" ;;
    FR) BIN=$B/intarsia-flip; ENVS="INTARSIA_BATCHED_MTP=1 INTARSIA_BATCHED_VERIFY=0 INTARSIA_VERIFY_TOPK=0"; ARGS="--mtp-draft-vocab 0" ;;
    *)  return 1 ;;
  esac
}
