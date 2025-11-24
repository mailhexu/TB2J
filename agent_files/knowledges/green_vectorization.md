# Green Function Vectorization Optimization

## Issue Fixed
Fixed `TypeError` in `exchange_pert2.py:86` where tuple division was attempted. The `pauli_block_all()` function returns a tuple of 4 arrays `(MI, Mx, My, Mz)`, not a single array. Changed to use `tuple((up + dn) / 2.0 for up, dn in zip(...))` for proper element-wise averaging.

## Vectorization Implementation

### Overview
Added vectorized versions of slow loop-heavy methods in `green.py` to improve performance. The original methods are kept for reference and backward compatibility.

### New Methods Added

#### 1. `get_dGR_vectorized()`
**Location**: `green.py:428-508`

**Purpose**: Vectorized computation of Green's function derivatives in real space.

**Key Optimizations**:
- Groups computations by `Rj` to minimize overhead
- Uses `np.einsum('nij,njk,nkl->il', ...)` to perform batched matrix multiplication
- Pre-allocates arrays for each `Rj` group
- Removes cutoff parameter (processes all Rk vectors)

**Formula**: `dG(Rj) = sum_{Rq,Rk} G(Rm) @ dV(Rq,Rk) @ G(Rnj)`

**Performance Benefit**: Replaces nested loops with batched einsum operations, significantly reducing Python overhead.

#### 2. `_build_Rmaps()`
**Location**: `green.py:510-540`

**Purpose**: Helper method to pre-build R-space mapping arrays for vectorization.

**What it does**:
- Builds `self._Rmap` for ij path
- Builds `self._Rmap_rev` for ji path
- Caches results to avoid recomputation
- Prints entry counts for debugging

#### 3. `get_GR_and_dGRdx_from_epw_vectorized()`
**Location**: `green.py:542-578`

**Purpose**: Vectorized version of `get_GR_and_dGRdx_from_epw` without cutoff.

**Key Changes**:
- Calls `get_dGR_vectorized()` instead of `get_dGR()`
- Removes `cutoff` parameter
- Uses same GR computation (already efficient)

**Returns**: `GR, dGRijdx, dGRjidx, rhoR`

### Usage

#### In exchange_pert2.py (line 149-152):

**Original (with cutoff)**:
```python
GR_up, dGRij_up, dGRji_up, rhoR_up = self.G.get_GR_and_dGRdx_from_epw(
    self.Rlist, self.short_Rlist, energy=e, epc=self.epc_up, Ru=self.Ru, cutoff=3.1)
```

**Vectorized (no cutoff, faster)**:
```python
GR_up, dGRij_up, dGRji_up, rhoR_up = self.G.get_GR_and_dGRdx_from_epw_vectorized(
    self.Rlist, self.short_Rlist, energy=e, epc=self.epc_up, Ru=self.Ru)
```

**Note**: 
- The vectorized version does **not** support the `cutoff` parameter
- If you need cutoff functionality, use the original method
- For best performance on large systems, use the vectorized version

### Technical Details

**Einsum Pattern**: `'nij,njk,nkl->il'`
- `n`: batch dimension (number of entries for each Rj)
- `ij, jk, kl`: matrix multiplication indices
- `il`: output matrix
- Automatically sums over the batch dimension `n`

**Memory Trade-off**: Pre-allocates larger arrays but reduces loop overhead significantly.

### Original Methods Kept
- `get_dGR()` - Original method with cutoff support at `green.py:351-426`
- `get_GR_and_dGRdx_from_epw()` - Original method at `green.py:324-349`

These are retained for:
1. Backward compatibility
2. Cases where cutoff is needed
3. Verification/testing purposes
