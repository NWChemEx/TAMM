

C From src/tce/tce_mo2e_offset_intorb.F
      INTEGER FUNCTION INDEX_PAIR(I,J)
      IMPLICIT NONE
      INTEGER I,J
      INDEX_PAIR=(I*(I-1))/2+J
      RETURN
      END       
