import CHalfKernels
import Foundation

/// y = bias + W x with W fp16 [m, n] row-major (n a multiple of 8), fp32 activations/accumulation.
/// Rows are split over `threads` GCD workers (1 = inline).
func gemvHalf(_ w: UnsafePointer<Float16>, _ x: UnsafePointer<Float>, _ bias: UnsafePointer<Float>,
              _ y: UnsafeMutablePointer<Float>, m: Int, n: Int, threads: Int) {
    precondition(n % 8 == 0, "gemvHalf needs n to be a multiple of 8")
    let raw = UnsafeRawPointer(w).assumingMemoryBound(to: UInt16.self)
    if threads <= 1 || m < 256 {
        eph_gemv_f16w(raw, x, bias, y, 0, Int32(m), Int32(n))
        return
    }
    let chunk = ((m + threads - 1) / threads + 3) & ~3
    DispatchQueue.concurrentPerform(iterations: threads) { i in
        let lo = i * chunk, hi = min(m, lo + chunk)
        if lo < hi { eph_gemv_f16w(raw, x, bias, y, Int32(lo), Int32(hi), Int32(n)) }
    }
}
