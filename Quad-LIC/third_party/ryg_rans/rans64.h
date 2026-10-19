











#ifndef RANS64_HEADER
#define RANS64_HEADER

#include <stdint.h>

#ifdef assert
#define Rans64Assert assert
#else
#define Rans64Assert(x)
#endif








#if defined(_MSC_VER)

#include <intrin.h>

static inline uint64_t Rans64MulHi(uint64_t a, uint64_t b)
{
    return __umulh(a, b);
}

#elif defined(__GNUC__)

static inline uint64_t Rans64MulHi(uint64_t a, uint64_t b)
{
    return (uint64_t) (((unsigned __int128)a * b) >> 64);
}

#else

#error Unknown/unsupported compiler!

#endif







#define RANS64_L (1ull << 31)  


typedef uint64_t Rans64State;


static inline void Rans64EncInit(Rans64State* r)
{
    *r = RANS64_L;
}








static inline void Rans64EncPut(Rans64State* r, uint32_t** pptr, uint32_t start, uint32_t freq, uint32_t scale_bits)
{
    Rans64Assert(freq != 0);

    
    uint64_t x = *r;
    uint64_t x_max = ((RANS64_L >> scale_bits) << 32) * freq; 
    if (x >= x_max) {
        *pptr -= 1;
        **pptr = (uint32_t) x;
        x >>= 32;
        Rans64Assert(x < x_max);
    }

    
    *r = ((x / freq) << scale_bits) + (x % freq) + start;
}


static inline void Rans64EncFlush(Rans64State* r, uint32_t** pptr)
{
    uint64_t x = *r;

    *pptr -= 2;
    (*pptr)[0] = (uint32_t) (x >> 0);
    (*pptr)[1] = (uint32_t) (x >> 32);
}



static inline void Rans64DecInit(Rans64State* r, uint32_t** pptr)
{
    uint64_t x;

    x  = (uint64_t) ((*pptr)[0]) << 0;
    x |= (uint64_t) ((*pptr)[1]) << 32;
    *pptr += 2;
    *r = x;
}


static inline uint32_t Rans64DecGet(Rans64State* r, uint32_t scale_bits)
{
    return *r & ((1u << scale_bits) - 1);
}




static inline void Rans64DecAdvance(Rans64State* r, uint32_t** pptr, uint32_t start, uint32_t freq, uint32_t scale_bits)
{
    uint64_t mask = (1ull << scale_bits) - 1;

    
    uint64_t x = *r;
    x = freq * (x >> scale_bits) + (x & mask) - start;

    
    if (x < RANS64_L) {
        x = (x << 32) | **pptr;
        *pptr += 1;
        Rans64Assert(x >= RANS64_L);
    }

    *r = x;
}









typedef struct {
    uint64_t rcp_freq;  
    uint32_t freq;      
    uint32_t bias;      
    uint32_t cmpl_freq; 
    uint32_t rcp_shift; 
} Rans64EncSymbol;


typedef struct {
    uint32_t start;     
    uint32_t freq;      
} Rans64DecSymbol;


static inline void Rans64EncSymbolInit(Rans64EncSymbol* s, uint32_t start, uint32_t freq, uint32_t scale_bits)
{
    Rans64Assert(scale_bits <= 31);
    Rans64Assert(start <= (1u << scale_bits));
    Rans64Assert(freq <= (1u << scale_bits) - start);

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    s->freq = freq;
    s->cmpl_freq = ((1 << scale_bits) - freq);
    if (freq < 2) {
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        s->rcp_freq = ~0ull;
        s->rcp_shift = 0;
        s->bias = start + (1 << scale_bits) - 1;
    } else {
        
        
        uint32_t shift = 0;
        uint64_t x0, x1, t0, t1;
        while (freq > (1u << shift))
            shift++;

        
        
        
        x0 = freq - 1;
        x1 = 1ull << (shift + 31);

        t1 = x1 / freq;
        x0 += (x1 % freq) << 32;
        t0 = x0 / freq;

        s->rcp_freq = t0 + (t1 << 32);
        s->rcp_shift = shift - 1;

        
        
        s->bias = start;
    }
}


static inline void Rans64DecSymbolInit(Rans64DecSymbol* s, uint32_t start, uint32_t freq)
{
    Rans64Assert(start <= (1 << 31));
    Rans64Assert(freq <= (1 << 31) - start);
    s->start = start;
    s->freq = freq;
}





static inline void Rans64EncPutSymbol(Rans64State* r, uint32_t** pptr, Rans64EncSymbol const* sym, uint32_t scale_bits)
{
    Rans64Assert(sym->freq != 0); 

    
    uint64_t x = *r;
    uint64_t x_max = ((RANS64_L >> scale_bits) << 32) * sym->freq; 
    if (x >= x_max) {
        *pptr -= 1;
        **pptr = (uint32_t) x;
        x >>= 32;
    }

    
    uint64_t q = Rans64MulHi(x, sym->rcp_freq) >> sym->rcp_shift;
    *r = x + sym->bias + q * sym->cmpl_freq;
}


static inline void Rans64DecAdvanceSymbol(Rans64State* r, uint32_t** pptr, Rans64DecSymbol const* sym, uint32_t scale_bits)
{
    Rans64DecAdvance(r, pptr, sym->start, sym->freq, scale_bits);
}




static inline void Rans64DecAdvanceStep(Rans64State* r, uint32_t start, uint32_t freq, uint32_t scale_bits)
{
    uint64_t mask = (1u << scale_bits) - 1;

    
    uint64_t x = *r;
    *r = freq * (x >> scale_bits) + (x & mask) - start;
}


static inline void Rans64DecAdvanceSymbolStep(Rans64State* r, Rans64DecSymbol const* sym, uint32_t scale_bits)
{
    Rans64DecAdvanceStep(r, sym->start, sym->freq, scale_bits);
}


static inline void Rans64DecRenorm(Rans64State* r, uint32_t** pptr)
{
    
    uint64_t x = *r;
    if (x < RANS64_L) {
        x = (x << 32) | **pptr;
        *pptr += 1;
        Rans64Assert(x >= RANS64_L);
    }

    *r = x;
}

#endif 
