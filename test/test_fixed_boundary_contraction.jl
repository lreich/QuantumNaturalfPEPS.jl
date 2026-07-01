using Test
using QuantumNaturalfPEPS
using QuantumNaturalGradient
using ITensors, ITensorMPS
using LinearAlgebra
using Random

Random.seed!(1234)

#=
    Tests for the fixed-boundary contraction routine (issue #4).

    The amplitude log(ψ(S)) is obtained by contracting the top boundary-MPS
    environment with the bottom one at some split position `pos`:

        get_logψ(env_top, env_down; pos)   ->   env_top[pos] · env_down[end-pos+1]

    Because the boundary-MPS is truncated to `contract_dim`, the result depends
    slightly on *where* we split. During the energy evaluation we reuse the
    environments and split at the flip row (cheap), so the same configuration S
    can end up with slightly different amplitudes ψ(S) depending on which row a
    flipped site sits in — this is what breaks the strict variational bound.

    The fixed-boundary routine always splits at a single, configuration-independent
    position (the lattice centre), so ψ(S) is one consistent value for a given S.
=#

@testset "Test: Fixed-boundary contraction" begin
    L = 4
    hilbert = siteinds("S=1/2", L, L)

    # Deliberately small contraction bond dimension so the boundary-MPS is
    # truncated and the split position actually matters.
    bond_dim = 2
    contract_dim = 2
    peps = PEPS(hilbert; bond_dim=bond_dim, contract_dim=contract_dim, sample_dim=contract_dim)

    S = rand([0, 1], size(peps))

    @testset "ψ(S) is consistent for the same configuration S" begin
        # Calling the fixed-boundary routine twice for the same S must give the
        # exact same amplitude: the split position is a deterministic function of
        # the system size only, never of the configuration.
        logψ1, env_top1, env_down1 = QuantumNaturalfPEPS.get_logψ_fixed_boundary(peps, S)
        logψ2, env_top2, env_down2 = QuantumNaturalfPEPS.get_logψ_fixed_boundary(peps, S)
        @test logψ1 == logψ2

        # Reusing the *same* environments and contracting them with the
        # fixed-boundary flag reproduces exactly the same ψ(S).
        logψ_reuse = QuantumNaturalfPEPS.get_logψ(env_top1, env_down1; fixed_boundary=true)
        @test logψ_reuse == logψ1

        # The fixed-boundary contraction is just the contraction at the canonical
        # centre split position.
        pos_fixed = QuantumNaturalfPEPS.fixed_boundary_pos(length(env_top1))
        @test 1 <= pos_fixed <= L - 1
        @test QuantumNaturalfPEPS.get_logψ(env_top1, env_down1; fixed_boundary=true) ==
              QuantumNaturalfPEPS.get_logψ(env_top1, env_down1; pos=pos_fixed)
    end

    @testset "ψ(S) changes slightly across different environment splits" begin
        # One set of environments for the configuration S ...
        _, env_top, env_down = QuantumNaturalfPEPS.get_logψ_fixed_boundary(peps, S)

        # ... contracted at every admissible split position. Each split pairs a
        # different env_top[pos] with a different env_down[end-pos+1], i.e. a
        # different pair of (truncated) environments for the *same* S.
        logψ_per_pos = [QuantumNaturalfPEPS.get_logψ(env_top, env_down; pos=p) for p in 1:L-1]

        max_dev = maximum(abs.(logψ_per_pos .- logψ_per_pos[1]))

        # The splits are NOT all bit-identical: the amplitude really does change
        # with the chosen environment split (the source of the non-variationality).
        @test !all(logψ_per_pos .== logψ_per_pos[1])
        @test max_dev > 0

        # ... but only *slightly*: they are all truncated approximations of the
        # same amplitude, so they stay close to each other.
        @test max_dev < 0.5

        # The fixed-boundary routine pins ψ(S) to a single one of these values,
        # independent of the split chosen anywhere else.
        pos_fixed = QuantumNaturalfPEPS.fixed_boundary_pos(length(env_top))
        @test QuantumNaturalfPEPS.get_logψ(env_top, env_down; fixed_boundary=true) ==
              logψ_per_pos[pos_fixed]
    end

    @testset "Exact contraction: every split agrees (truncation artifact only)" begin
        # With a large enough contraction dimension there is no truncation, so the
        # amplitude becomes split-independent. This confirms that the deviation
        # seen above is purely a truncation artifact, not a bug in the contraction.
        exact_dim = 2^(L * L)
        peps_exact = PEPS(hilbert; bond_dim=bond_dim, contract_dim=exact_dim, sample_dim=exact_dim)
        QuantumNaturalfPEPS.set_params!(peps_exact, peps.tensors) # reuse the same tensors

        logψ_fixed, env_top, env_down = QuantumNaturalfPEPS.get_logψ_fixed_boundary(peps_exact, S)
        logψ_per_pos = [QuantumNaturalfPEPS.get_logψ(env_top, env_down; pos=p) for p in 1:L-1]

        @test all(isapprox.(logψ_per_pos, logψ_per_pos[1]; atol=1e-8))

        # ... and it matches the exact dense contraction of the projected PEPS.
        logψ_dense = QuantumNaturalfPEPS.logψ_exact(peps_exact, S)
        @test isapprox(logψ_fixed, logψ_dense; rtol=1e-8)
    end

    @testset "Local energy via get_Ek(...; fixed_boundary=true)" begin
        ham = QuantumNaturalfPEPS.hamiltonain_J1J2(1.0, 0.5, L, L)
        ham_op = QuantumNaturalfPEPS.TensorOperatorSum(ham, hilbert)
        S_e = rand([0, 1], size(peps))

        # The fixed-boundary energy must equal the energy obtained from an
        # independent reference that feeds the fixed-boundary amplitude function to
        # QuantumNaturalGradient.get_Ek (which recomputes ψ(S') for every flip).
        func_fixed(sample) = first(QuantumNaturalfPEPS.get_logψ_fixed_boundary(peps, sample))
        E_ref = QuantumNaturalfPEPS.convert_if_real(QuantumNaturalGradient.get_Ek(S_e, ham_op, func_fixed))
        E_fixed = QuantumNaturalfPEPS.get_Ek(peps, ham_op, S_e; fixed_boundary=true)
        @test isapprox(E_fixed, E_ref; rtol=1e-10)

        # Under truncation it must differ from the cheap reuse-the-environment
        # energy: the two contraction schemes disagree, which is exactly the
        # non-variational artifact issue #4 is about.
        E_fast = QuantumNaturalfPEPS.get_Ek(peps, ham_op, S_e) # default fixed_boundary=false
        @test !isapprox(E_fixed, E_fast; rtol=1e-8)
    end

    @testset "Exact contraction: fixed and fast energies agree" begin
        ham = QuantumNaturalfPEPS.hamiltonain_J1J2(1.0, 0.5, L, L)
        ham_op = QuantumNaturalfPEPS.TensorOperatorSum(ham, hilbert)

        exact_dim = 2^(L * L)
        peps_exact = PEPS(hilbert; bond_dim=bond_dim, contract_dim=exact_dim, sample_dim=exact_dim)
        QuantumNaturalfPEPS.set_params!(peps_exact, peps.tensors)

        S_e = rand([0, 1], size(peps_exact))
        E_fast = QuantumNaturalfPEPS.get_Ek(peps_exact, ham_op, S_e)
        E_fixed = QuantumNaturalfPEPS.get_Ek(peps_exact, ham_op, S_e; fixed_boundary=true)
        E_exact = QuantumNaturalfPEPS.convert_if_real(
            QuantumNaturalGradient.get_Ek(S_e, ham_op, s -> QuantumNaturalfPEPS.logψ_exact(peps_exact, s)))

        # With no truncation both contraction schemes reproduce the exact energy.
        @test isapprox(E_fast, E_exact; rtol=1e-8)
        @test isapprox(E_fixed, E_exact; rtol=1e-8)
    end
end;
