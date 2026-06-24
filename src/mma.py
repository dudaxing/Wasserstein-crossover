"""
Method of Moving Asymptotes (MMA) -- Svanberg's algorithm.

Compact, self-contained NumPy implementation of the MMA subproblem solver used
for the low-fidelity (gradient-based) topology optimization in the paper
(Section 4.1: "We use the method of moving asymptotes (MMA) [68] for updating
the design variables").

This follows the standard formulation:
    K. Svanberg, "The method of moving asymptotes - a new method for structural
    optimization", Int. J. Numer. Methods Eng. 24 (1987) 359-373,
and the publicly documented matrix form of `mmasub`/`subsolv`.

Solves, at each design iteration, the nonlinear program
    min  f0(x) + a0*z + sum( c_i*y_i + 0.5*d_i*y_i^2 )
    s.t. f_i(x) - a_i*z - y_i <= 0,   i=1..m
         xmin <= x <= xmax,  y >= 0,  z >= 0
by building convex MMA approximations and solving the dual with a primal-dual
interior point method (`subsolv`).
"""
from __future__ import annotations
import numpy as np


def mmasub(m, n, itr, xval, xmin, xmax, xold1, xold2,
           f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d,
           move=0.5, asyinit=0.5, asyincr=1.2, asydecr=0.7):
    """One MMA design step.

    All vector inputs are column vectors of shape (n,1) or (m,1).
    Returns (xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp).
    """
    epsimin = 1e-7
    one = np.ones((n, 1))
    onem = np.ones((m, 1))
    xval = xval.reshape(-1, 1)

    # --- update asymptotes low/upp -----------------------------------------
    if itr <= 2:
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = np.ones((n, 1))
        factor[zzz > 0] = asyincr
        factor[zzz < 0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        lowmin = xval - 10.0 * (xmax - xmin)
        lowmax = xval - 0.01 * (xmax - xmin)
        uppmin = xval + 0.01 * (xmax - xmin)
        uppmax = xval + 10.0 * (xmax - xmin)
        low = np.maximum(low, lowmin)
        low = np.minimum(low, lowmax)
        upp = np.minimum(upp, uppmax)
        upp = np.maximum(upp, uppmin)

    # --- bounds for the subproblem (alpha, beta) ---------------------------
    zzz1 = low + 0.1 * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    alfa = np.maximum.reduce([zzz1, zzz2, xmin])
    zzz1 = upp - 0.1 * (upp - xval)
    zzz2 = xval + move * (xmax - xmin)
    beta = np.minimum.reduce([zzz1, zzz2, xmax])

    # --- build p, q coefficients -------------------------------------------
    xmami = np.maximum(xmax - xmin, 1e-5 * one)
    ux1 = upp - xval
    xl1 = xval - low
    ux2 = ux1 ** 2
    xl2 = xl1 ** 2
    df0dx = df0dx.reshape(-1, 1)
    p0 = np.maximum(df0dx, 0.0)
    q0 = np.maximum(-df0dx, 0.0)
    pq0 = 0.001 * (p0 + q0) + 1e-5 / xmami
    p0 = (p0 + pq0) * ux2
    q0 = (q0 + pq0) * xl2

    dfdx = dfdx.reshape(m, n)
    P = np.maximum(dfdx, 0.0)
    Q = np.maximum(-dfdx, 0.0)
    PQ = 0.001 * (P + Q) + 1e-5 / xmami.T
    P = (P + PQ) * (ux2.T)
    Q = (Q + PQ) * (xl2.T)
    b = (P @ (1.0 / ux1) + Q @ (1.0 / xl1)) - fval.reshape(m, 1)

    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(
        m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q,
        a0, a.reshape(m, 1), b, c.reshape(m, 1), d.reshape(m, 1))
    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp


def subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d):
    """Primal-dual interior point solver for the MMA subproblem."""
    een = np.ones((n, 1))
    eem = np.ones((m, 1))
    epsi = 1.0
    x = 0.5 * (alfa + beta)
    y = eem.copy()
    z = np.array([[1.0]])
    lam = eem.copy()
    xsi = np.maximum(1.0 / (x - alfa), een)
    eta = np.maximum(1.0 / (beta - x), een)
    mu = np.maximum(eem, 0.5 * c)
    zet = np.array([[1.0]])
    s = eem.copy()

    while epsi > epsimin:
        epsvecn = epsi * een
        epsvecm = epsi * eem
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 ** 2
        xl2 = xl1 ** 2
        plam = p0 + P.T @ lam
        qlam = q0 + Q.T @ lam
        gvec = P @ (1.0 / ux1) + Q @ (1.0 / xl1)
        dpsidx = plam / ux2 - qlam / xl2

        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - a.T @ lam
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm
        rezet = zet * z - epsi
        res = lam * s - epsvecm

        residu = np.concatenate(
            (rex, rey, rez, relam, rexsi, reeta, remu, rezet, res))
        residunorm = np.sqrt(float(np.dot(residu.ravel(), residu.ravel())))
        residumax = float(np.max(np.abs(residu)))

        itt = 0
        while residumax > 0.9 * epsi and itt < 200:
            itt += 1
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 ** 2
            xl2 = xl1 ** 2
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            plam = p0 + P.T @ lam
            qlam = q0 + Q.T @ lam
            gvec = P @ (1.0 / ux1) + Q @ (1.0 / xl1)
            GG = P / ux2.T - Q / xl2.T
            dpsidx = plam / ux2 - qlam / xl2

            delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
            dely = c + d * y - lam - epsvecm / y
            delz = a0 - a.T @ lam - epsi / z
            dellam = gvec - a * z - y - b + epsvecm / lam

            diagx = plam / ux3 + qlam / xl3
            diagx = 2.0 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = 1.0 / diagx
            diagy = d + mu / y
            diagyinv = 1.0 / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv

            if m < n:
                blam = dellam + dely / diagy - GG @ (delx / diagx)
                bb = np.concatenate((blam, delz), axis=0)
                Alam = np.asarray(np.diag(diaglamyi.flatten())) + \
                    (GG * diagxinv.T) @ GG.T
                AA = np.zeros((m + 1, m + 1))
                AA[:m, :m] = Alam
                AA[:m, m:m + 1] = a
                AA[m:m + 1, :m] = a.T
                AA[m, m] = float((-zet / z).item())
                solut = np.linalg.solve(AA, bb)
                dlam = solut[:m]
                dz = solut[m:m + 1]
                dx = -delx / diagx - (GG.T @ dlam) / diagx
            else:
                diaglamyiinv = 1.0 / diaglamyi
                dellamyi = dellam + dely / diagy
                Axx = np.asarray(np.diag(diagx.flatten())) + \
                    (GG.T * diaglamyiinv.T) @ GG
                azz = zet / z + a.T @ (a / diaglamyi)
                axz = -GG.T @ (a / diaglamyi)
                bx = delx + GG.T @ (dellamyi / diaglamyi)
                bz = delz - a.T @ (dellamyi / diaglamyi)
                AA = np.zeros((n + 1, n + 1))
                AA[:n, :n] = Axx
                AA[:n, n:n + 1] = axz
                AA[n:n + 1, :n] = axz.T
                AA[n, n] = float(np.asarray(azz).item())
                bb = np.concatenate((-bx, -bz), axis=0)
                solut = np.linalg.solve(AA, bb)
                dx = solut[:n]
                dz = solut[n:n + 1]
                dlam = (GG @ dx) / diaglamyi - dz * (a / diaglamyi) + \
                    dellamyi / diaglamyi

            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsvecm / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsvecm / lam - (s * dlam) / lam

            xx = np.concatenate((y, z, lam, xsi, eta, mu, zet, s))
            dxx = np.concatenate((dy, dz, dlam, dxsi, deta, dmu, dzet, ds))
            stepxx = -1.01 * dxx / xx
            stmxx = float(np.max(stepxx))
            stepalfa = -1.01 * dx / (x - alfa)
            stmalfa = float(np.max(stepalfa))
            stepbeta = 1.01 * dx / (beta - x)
            stmbeta = float(np.max(stepbeta))
            stmalbe = max(stmalfa, stmbeta)
            stmalbexx = max(stmalbe, stmxx)
            stminv = max(stmalbexx, 1.0)
            steg = 1.0 / stminv

            xold = x.copy(); yold = y.copy(); zold = z.copy()
            lamold = lam.copy(); xsiold = xsi.copy(); etaold = eta.copy()
            muold = mu.copy(); zetold = zet.copy(); sold = s.copy()

            itto = 0
            resinew = 2.0 * residunorm
            while resinew > residunorm and itto < 50:
                itto += 1
                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds
                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1 ** 2
                xl2 = xl1 ** 2
                plam = p0 + P.T @ lam
                qlam = q0 + Q.T @ lam
                gvec = P @ (1.0 / ux1) + Q @ (1.0 / xl1)
                dpsidx = plam / ux2 - qlam / xl2

                rex = dpsidx - xsi + eta
                rey = c + d * y - mu - lam
                rez = a0 - zet - a.T @ lam
                relam = gvec - a * z - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm
                rezet = zet * z - epsi
                res = lam * s - epsvecm
                residu = np.concatenate(
                    (rex, rey, rez, relam, rexsi, reeta, remu, rezet, res))
                resinew = np.sqrt(float(np.dot(residu.ravel(), residu.ravel())))
                steg *= 0.5
            residunorm = resinew
            residumax = float(np.max(np.abs(residu)))
            steg *= 2.0
        epsi *= 0.1

    return x, y, z, lam, xsi, eta, mu, zet, s
