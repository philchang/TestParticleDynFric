
module orbit_integrator

  double precision, parameter :: Msun=1.9891D33, Gn=6.67348D-8
  double precision, parameter :: lengthscale = 3D21, softening = 1D-3*lengthscale
  double precision, parameter :: kpc = 3D21, kms=1D5, Gyr = 1D9*3.15D7

! mass0=1.03D12*Msun
  double precision :: mass0 = 1.29d12*Msun, &  ! this is M200 of Springel & White 1999, or M_HALO in structure.c
       scalelength0 = 20D0*kpc, &
       v0200 = 1.8D7, c0 = 9.39,  &  ! for hernquist 
       ! NFW parameters (K13 values)
!        Rvir = 329D0*kpc, cvir = 9.36, Mvir = 2.d12*Msun, anfw
       Rvir = 299D0*kpc, cvir = 9.56, Mvir = 1.5d12*Msun, anfw, &
       V0sis = 2.2D7, masssis = 1d12*Msun  !for SIS

  integer, parameter :: PROFILE_TYPE = 3 ! 1 for SIS, 2 for NFW, 3 for Hernquist profile that has same dm mass within R200 as NFW
  double precision, parameter :: pi = 0.5D0*6.28318530717959d0
  double precision, parameter :: rin = 5D0*kpc, rout = 60D0*kpc

  integer :: numPerturbers
  integer :: iter = 100
  integer, parameter :: MAX_PERTURBERS = 200
  double precision, dimension(MAX_PERTURBERS) :: massp
  double precision, dimension(MAX_PERTURBERS) :: scalelengthp
  double precision, dimension(MAX_PERTURBERS) :: cp
  double precision, dimension(MAX_PERTURBERS) :: vp200

  double precision, dimension(MAX_PERTURBERS) :: xp0, yp0, zp0, vxp0, vyp0, vzp0
  double precision, dimension(MAX_PERTURBERS) :: periDistance
  double precision, dimension(MAX_PERTURBERS) :: periTime

  double precision, dimension(MAX_PERTURBERS,3) :: posp_start ! starting x,y,z of perturbers
  double precision, dimension(MAX_PERTURBERS,3) :: vposp_start ! starting vx,vy,vz of perturbers

  double precision, dimension(MAX_PERTURBERS,3) :: posp ! x,y,z of perturbers
  double precision, dimension(MAX_PERTURBERS,3) :: vposp ! vx,vy,vz of perturbers

  ! FOURIER MODES  
  integer, parameter :: NUMMODES = 4 ! up to m=5
  double precision, parameter :: fourier_rin = 10D0*kpc, fourier_rout=45D0*kpc ! this sets the regions where we calculate fourier modes

  logical, parameter :: INTEGRATE_BACKWARD_FIRST = .false.  ! set this to true if you want to integrate 
                                                            ! from present position backwards to get initial conditions

end module orbit_integrator

module rings
  integer, parameter :: numradial=500 ! for the particle positions
  integer, parameter :: numradials=numradial ! for the fourier rings
end module rings

module errorFunctionStuff
  integer, parameter :: NUM_INTERVALS = 5000
  double precision :: ERROR_HI = 5D0
  double precision, dimension(NUM_INTERVALS) :: errorFunctionCache
  logical :: errorFunctionCacheInitialized = .false.
end module errorFunctionStuff

program nbody
  use rings
  use orbit_integrator
  implicit none

  !integer, parameter :: numparts = 1000!, numradial=200  
  integer, parameter :: numparts = 100000
  double precision, dimension (numparts,6) :: rparts
  double precision :: tau, dtau, dstau, tdyn, tau_end, t1
  double precision, dimension(3) :: r, v, zvec, xvec, yvec, rpvec, rg, sinv, rh
  double precision :: t, e, semi
  real :: harvest
  integer :: i, ndyns, k,j, stp
  integer :: steps = 50
  double precision :: v0, v1
  double precision :: absdeltar, absr, dadr, dadphi, absrg
  double precision, dimension(3) :: acc, deltar, acc0, dv
  double precision, parameter :: frac = 0.2D0
  double precision :: mass, rho, sigma
  double precision :: dvr, dvphi, dvx
  double precision, parameter :: dvr0 = 4.858D6, dvphi0 = -2.899D6 
  external :: integrate_orbit, fourierdata, update_perturber_positions, getprofile
  integer, parameter :: ith = 2
  double precision :: startingTime
  double precision :: theta, absacc, sep, x1, y1
  double precision :: accx, accy, vp, rp, Omega

  double precision :: accr, accphi, accr1, accphi1
  double precision :: dr, dr2, dphi

  double precision :: r0
  double precision :: stau

  double precision :: rcross, vrcross, jcross, vzcross, vphicross
  double precision :: e0, dradius, radius
  double precision, dimension(MAX_PERTURBERS,6) :: rstartp
  double precision :: thetap, phip, rclose
  
  double precision :: distance
  integer :: numRealizations, iRealization

  double precision, dimension( NUMMODES) :: amtot
  double precision :: a0tot, ateff
  
  character(len=80) :: filename, realization_filename
  ! initialize the file
  call getarg(1, filename)
  call initialize_perturber_file(filename)
  write(*,*) "Read initial data"
  ! reset the pericenter
  periDistance = 999D0*kpc

if (PROFILE_TYPE == 1)  then
  vp200 = V0sis*(massp/masssis)**0.333D0  ! for SIS
endif

if (PROFILE_TYPE == 2)  then 
  vp200 = v0200*(massp/Mvir)**0.333D0     ! for NFW
endif

if (PROFILE_TYPE == 3)  then
  vp200 = v0200*(massp/mass0)**0.333D0    ! for Hernquist profile w. same DM within R200 as NFW and same inner slope
endif

  ndyns = 40D0/frac
  semi = 10D0*lengthscale! + lengthscale*floor(1D0*numradial*i/numparts)/numradial
  ! semi is used below to determine the dynamical time for integration purposes so changing it changes the timestep
  ! this should be changed to adaptive time step in the future.

   call getprofile(semi, mass, rho, sigma)
!   call getprofile(180.*kpc,mass,rho,sigma)

!   mass = hernquistmass(semi, mass0, v0200, c0)
!  mass = hernquistmass(300D0*lengthscale, 1.75d12*Msun, v0200, c0)
!  mass = nfwmass(anfw,mass0,299.*kpc)
!  write (*,*) 'mass0,v0200,c0,mass=',mass0/1.9891D33,v0200,c0,(mass/1.9891D33)/1.d12
!   write (*,*) 'mass out to 180 kpc=',(mass/1.989d33)/1.d12
!   pause

  tdyn = frac*sqrt(semi**3D0/(Gn*mass))
  tau_end = 2.65D0*Gyr!ndyns*tdyn
  
  ! get the starting conditions for the perturber
  startingTime = 2.65D0*Gyr ! integrate backward by this time
  call get_starting_conditions(-startingTime, rstartp)  ! integrate backwards 
  
  rparts(1:numPerturbers,:) = rstartp(1:numPerturbers,:) ! set the starting conditions

  call update_perturber_positions(numparts, rparts)
  
  do i = 1, numPerturbers
      write(*,fmt='(1x,6(D15.5))') posp(i,1)/kpc, posp(i,2)/kpc, posp(i,3)/kpc, vposp(i,1)/kms, vposp(i,2)/kms, vposp(i,3)/kms
  end do

  open (78,file='tau_vs_ateff.dat',status='unknown')

  !! Store                                                                                                                                                  
      open (12,file='outputBack.dat',status='unknown')
      do i = 1, numPerturbers
         write (12,fmt='(1x,9(D15.5))') massp(i)/Msun,periTime(i)/3.15D13, periDistance(i)/kpc, posp(i,:)/kpc, vposp(i,:)/kms
      end do
      close (12)

  !!pause ' '
   
  ! set the starting condition for later
  posp_start = posp/kpc
  vposp_start = vposp/kms

  ! set uo disk particles
  do k = 1, numradial
     semi = rin + (rout-rin)*(k-1)/numradial

     call getprofile(semi, mass, rho, sigma)
     
     v0 = sqrt(Gn*mass/(semi)) 
     
     !     Jm(1,1,k) = 0.1D0*v0*semi
     do j = 1, numparts/numradial
        
        i = numPerturbers+(k-1)*numparts/numradial + j
        
        theta = mod(6.282D0*j/(numparts/numradial),6.282D0)
        
        v1 = v0*(1D0+0.1D0*sin(theta))
        
        theta = theta !+ 0.2D0*cos(theta)
        if(i<=numparts) then
           rparts(i,1) = semi!*(1D0-0.1D0*cos(1D0*theta))
           rparts(i,2) = 0D0
           rparts(i,3) = v0*semi!-0.1D0*v0*semi*cos(theta*2D0)
           rparts(i,4) = theta
           rparts(i,5) = 0D0
           rparts(i,6) = 0D0
           
        end if
     end do
  end do

! writing current time in Myr and X,Y,Z of satellites: 
  open (21,file='XYZSat1.dat',status='unknown')
!  open (22,file='XYZSat2.dat',status='unknown')
!  open (23,file='XYZSat3.dat',status='unknown')

  dtau = tdyn
  tau = 0d0
  do while (tau < tau_end) 
!  do tau = 0D0, tau_end, tdyn

     write(*,fmt='(A40,F9.2)') "Current run time [Myrs]=", tau/3.15D13       

     stau = tau
     dstau = dtau/steps
     do stp = 1, steps
        call integrate_orbit(stau, dstau, numparts, rparts)
        stau = stau + dstau
     enddo

     ! update time and timestep
     tau = tau + dtau
     
     if( tau+dtau > tau_end) dtau = tau_end-tau ! integrate exactly to the end

     ! get the peridistance
     do i = 1, numPerturbers 
        distance = sqrt(rparts(i,1)**2. + rparts(i,5)**2) ! r^2 + z^2
        if( distance < periDistance(i)) then 
            periDistance(i) = distance
            periTime(i)=tau
        endif
     end do

     write(21,*) tau/3.15D13, posp(1,:)/kpc, vposp(1,:)/kms
!     write(22,*) tau/3.15D13, posp(2,:)/kpc, vposp(2,:)/kms
!     write(23,*) tau/3.15D13, posp(3,:)/kpc, vposp(3,:)/kms

     call writedata(tau/Gyr, numPerturbers, numparts, rparts,iter)
     
     call fourierdata(tau/Gyr, numparts, rparts, amtot, a0tot, ateff,iter)
     iter = iter +1
!     write( realization_filename, fmt='(A14,E9.3,A8)') "fourierm1m5_t=", tau/Gyr, "Gyr.data"
     write(78,fmt="(15(2X,E10.3))") tau/Gyr, ateff, sqrt(rparts(1:numPerturbers,1)**2. + rparts(1:numPerturbers,5)**2)/kpc
!, & 
!          sqrt(rparts(2,1)**2. + rparts(2,5)**2)/kpc, periDistance(2)/kpc, sqrt(rparts(3,1)**2. + rparts(3,5)**2)/kpc, periDistance(3)/kpc
     !call write_out_realization( realization_filename, amtot, a0tot)
  end do
 
  close(21)   ! the XYZ files
!  close(22)
!  close(23)

  call write_out_realization( "fourierm1m5.dat", amtot, a0tot)

  close(78)
!     write(*,fmt='(9(2X,D12.6))') t/tdyn, l(1), l(2), l(3), (energy-energy0)/energy0, acos(dot_product(l, n1))/pi*180D0, acos(dot_product(l, nCW))/pi*180D0, eccent, massenc
  stop

end program nbody

subroutine initialize_perturber_file( filename) 
  use orbit_integrator
  implicit none
  character(len=80), intent(IN) :: filename
  integer :: i
  double precision :: r, v
  double precision :: concMaccio
  external :: concMaccio
  
  open(unit = 37, file = filename, status = 'old')

  read(unit=37,fmt='(1X,I8)') numPerturbers
  write(*,*),'numPerturbers=',numPerturbers

  if( numPerturbers > MAX_PERTURBERS) then
     write(*,*) "ERROR: Increase MAX_PERTURBERS"
     stop
  end if

  xp0 = 0D0
  yp0 = 0D0
  zp0 = 0D0

  vxp0 = 0D0
  vyp0 = 0D0
  vzp0 = 0D0

  do i = 1, numPerturbers
     read(unit=37,fmt='(1x,20(E15.5))') massp(i), r, v, xp0(i), yp0(i), zp0(i), vxp0(i), vyp0(i), vzp0(i)  
     cp(i) = concMaccio(massp(i))
  end do

  massp = massp*Msun
  xp0 = xp0*kpc
  yp0 = yp0*kpc
  zp0 = zp0*kpc

  vxp0 = vxp0*kms
  vyp0 = vyp0*kms
  vzp0 = vzp0*kms
  
  return
end subroutine initialize_perturber_file

subroutine close_perturber_file
  implicit none
  close( unit = 37)
  return
end subroutine close_perturber_file


subroutine write_out_realization( filename, amtot, a0tot)
  use orbit_integrator
  implicit none
  
  double precision, dimension( NUMMODES) :: amtot
  double precision :: a0tot
  character(LEN=40) :: filename
  integer :: i 
  ! write out the fourier response and pericenter distance
  ! m = 1-5 amplitudes and phase

  open (13,file=filename,status='unknown')
  write(13, fmt='(10(2X,E9.2))') amtot/a0tot!, atan(amsintot/amcostot)
  close(13)

  write(*,*) 'number of perturbers'
  write(*,fmt='(I8)') numPerturbers

  write(*,*) 'mass, peridistance, periTime,x,y,z, vx,vy, vz'
  do i = 1, numPerturbers
     write(*,fmt='(1x,9(D15.5))') massp(i)/Msun, periDistance(i)/kpc, periTime(i)/3.15D13,posp(i,:)/kpc, vposp(i,:)/kms
  end do

!! Store                                                                                                          
      open (12,file='output.dat',status='unknown')
      do i = 1, numPerturbers
         write (12,fmt='(1x,9(D15.5))') massp(i)/Msun, periTime(i)/3.15D13, periDistance(i)/kpc, posp(i,:)/kpc, vposp(i,:)/kms
      end do
      close (12)
  return
end subroutine write_out_realization

subroutine get_starting_conditions(startingTime, rstartp)
  use rings
  use orbit_integrator
  implicit none
  double precision, intent(IN) :: startingTime
  double precision, intent(OUT), dimension(MAX_PERTURBERS,6) :: rstartp
  double precision, dimension(numPerturbers,6) :: rpertu, rpertuout
  double precision, dimension(numPerturbers) :: theta
  double precision :: tau, absr, mass, tdyn, t, rho, sigma
  double precision :: hernquistmass, dot_product
  external :: hernquistmass, dot_product, integrate_io_rk,getprofile
  integer :: i

  integer, parameter :: fine = 2
  
  rpertu(:,1) = sqrt(xp0(1:numPerturbers)**2.0 + yp0(1:numPerturbers)**2.0) ! setup r
  theta = atan2(yp0(1:numPerturbers),xp0(1:numPerturbers)) ! setup theta
  rpertu(:,2) = cos(theta)*vxp0(1:numPerturbers) + sin(theta)*vyp0(1:numPerturbers) ! setup vr
  rpertu(:,3) = xp0(1:numPerturbers)*vyp0(1:numPerturbers) - yp0(1:numPerturbers)*vxp0(1:numPerturbers) ! setup angular momentum
  rpertu(:,4) = theta(:)
  rpertu(:,5) = zp0(1:numPerturbers)
  rpertu(:,6) = vzp0(1:numPerturbers)

  absr = sqrt(rpertu(1,1)**2D0 + rpertu(1,5)**2D0)
  call getprofile( absr, mass, rho, sigma)
  tdyn = sqrt(absr*absr*absr/(Gn*mass))
  tdyn = 1D5*3.15e7   
  
  if( INTEGRATE_BACKWARD_FIRST) then
     tau = -tdyn ! run movie backwards in time
  
     t = 0D0

     do while(t > startingTime) 
        call integrate_io_rk(0,numPerturbers,tau,fine,rpertu,rpertuout)
        rpertu = rpertuout
        t = t + tau
        if( t+tau < startingTime) tau = t-startingTime
     end do
  end if

  rstartp(1:numPerturbers,:) = rpertu(:,:)
  return
end subroutine get_starting_conditions


subroutine fourierdata(tau, numparts, rparts, amtot, a0tot, ateff, num)
  use rings
  use orbit_integrator

  implicit none
  integer :: numparts, i, k, tindex, rindex, m, num
  double precision :: tau, dphi, a0, norma0
  double precision, dimension(NUMMODES) :: amcos, amsin
  double precision, dimension(NUMMODES) :: amcostot, amsintot
  double precision, dimension(NUMMODES) :: amtot
  double precision :: ateff
  double precision :: a0tot
  double precision, dimension(numparts,6) :: rparts
  double precision :: r, theta
  
  double precision :: v2, energy
  double precision, dimension(numparts) :: eps, semi

  integer, parameter :: numphi = 128
  double precision, dimension( numradials, numphi) :: dens
  double precision, dimension( 2*numphi) :: denstemp
  integer :: total_particles = 0
  double precision :: diskprofile, rann
  character(LEN=40) :: filename


  external :: dfour1

  write( filename, fmt='(A12,I3,A4)') "fourier_data",num,".txt"
  open(unit=25, file=filename, status='new')  

!  write( filename, fmt='(A15,E9.3,A8)') "fourier_data_t=", tau, "Gyr.data"
!  open(unit=25, file=filename, status='new')
  

! write(filename,fmt='(A8,E9.3,A8)') "Sigma_t=", tau, "Gyr.data"
! open(unit=26, file=filename, status='new')

  total_particles = 0
  i = 1
  r = rparts(i,1)

  !write(25,fmt='(F4.1)') r/3D21

  ! calculate the density
  dens = 0D0

  do i = 2, numparts
     r = rparts(i,1)
     theta = mod(rparts(i,4)+4D0*pi, 2D0*pi)

     tindex = int(floor(theta/(2D0*pi)*numphi)) + 1
     if( r > rin .and. r <= rout) then
        rindex = int(floor((r - rin +0.5*kpc)/(rout-rin)*numradials)) + 1
     
        if( rindex > 0 .and. rindex <= numradials .and. tindex > 0 .and. tindex <= numphi) then
           dens(rindex, tindex) = dens(rindex, tindex) + 1
        end if
     end if
  end do

  do i = 1, numradials
    rann = rin + (rout-rin)*(1D0*(i-1)/numradials + 0.5D0/numradials)
!    write(26, fmt='(129(2X,E16.6))') rann, dens(i,:)
!    print *,'rann=',rann
!    stop
 end do

!  close(unit=26)
  
  a0tot = 0D0
  amtot = 0D0

  do i = 1, numradials
!  i = numradials/2
     amcos = 0D0
     amsin = 0D0
     a0 = 0D0
     dphi = 1D0/numphi

     norma0 = numparts/numradials

     do k = 1, numphi
        a0 = a0 + dens(i,k)*dphi
     end do
     
     if(a0 < 0D0) a0 = 0D0

     total_particles = 0
     denstemp = 0D0
     do k = 1, numphi
        denstemp(2*k-1) = dens(i,k)
        total_particles = total_particles + dens(i,k)
     end do
     call dfour1(denstemp, numphi, 1)
     denstemp = denstemp/(numphi)

     do m = 1, NUMMODES
        amcos(m) = denstemp(2*m + 1)
        amsin(m) = denstemp(2*m + 2)
     end do

     rann = rin + (rout-rin)*(1D0*(i-1)/numradials + 0.5D0/numradials)
     diskprofile = 1D0 !exp(-rann/diskscale)

     if( rann > fourier_rin .and. rann < fourier_rout) then
        a0tot = a0tot + diskprofile*a0
        amtot = amtot + diskprofile*sqrt(amcos*amcos + amsin*amsin)

     write(25,fmt='(10(2X,E13.6),2X,I8)') rann, & 
              sqrt(amcos**2D0 + amsin**2D0)/(norma0*dphi), a0/(norma0*dphi), atan(amsin/amcos), total_particles

     end if

  end do

  close(unit=25)

  ateff = 0D0
  do i = 1, NUMMODES
     ateff = ateff + (amtot(i)/a0tot)**2D0
  end do
  ateff = sqrt(ateff/NUMMODES)

  return 
end subroutine fourierdata

subroutine integrate_orbit(stau, tau, numparts, rparts)
  use rings
  use orbit_integrator
  implicit none

  integer, intent(in) :: numparts
  integer :: i, j, k
  integer :: fine
  
  double precision, dimension(numparts, 6), intent(inout) :: rparts
  double precision, dimension(numparts, 6) :: rpartsout
  double precision, dimension(1, 6) :: rperturber


  double precision, dimension(3) :: r, deltar, acc
  double precision :: rp, thetap
  double precision, intent(in) :: tau, stau
  external :: integrate_io_rk

  fine = 2

  call integrate_io_rk(0, numparts,tau,fine,rparts,rpartsout) ! full
  rparts = rpartsout
  
  return

end subroutine integrate_orbit


subroutine integrate_io_rk(gc, numparts,tau,fine,rparts,rpartsout)     
  use orbit_integrator
  implicit none

  integer, intent(in) :: numparts, fine, gc
  integer, parameter :: noeqs = 7
  double precision, dimension(numparts, 6), intent(in) :: rparts
  double precision, dimension(numparts, 6), intent(OUT) :: rpartsout
  double precision, dimension(noeqs) :: y, yout, yerr, height

  integer :: i, j, k, m, p

  !
  ! Sum over all particles
  !

  double precision :: r, vr, angmom, theta

  double precision, intent(in) :: tau
  double precision :: testenergy, mass, absdeltar, dt, absr 
  
  double precision :: nfwmass, hernquistmass
  external :: nfwmass, hernquistmass, derivs, rkckstep, update_perturber_positions

  double precision :: z, vz, zp, vzp
  
  if(abs(tau) < 1D-20) return

 
  dt = tau/fine

  rpartsout = rparts
        
  do m = 1, fine
     !$OMP PARALLEL DO PRIVATE(i,r,vr,angmom,theta,z,vz,y,yout,height,yerr)
     do i = 1, numparts
 
        r = rpartsout(i,1)
        vr = rpartsout(i,2)
        angmom = rpartsout(i,3)
        theta = rpartsout(i,4)
        z = rpartsout(i,5)
        vz = rpartsout(i,6)

        y(1) = r
        y(2) = vr
        y(3) = angmom
        y(4) = theta

        y(5) = z
        y(6) = vz

        y(7) = 1D0*i
        
        if ( gc == 1 ) y(7) = -1D0

        call rkckstep(noeqs, 0D0, dt, y, height, yout, yerr,derivs)

        rpartsout(i,1:4) = yout(1:4)
        rpartsout(i,5:6) = yout(5:6)
        
     end do
     !$OMP END PARALLEL DO

     ! update perturber positions
     call update_perturber_positions( numparts, rpartsout)
  end do

!  rpartsout = rparts

!  print *,'in integrate_io_rk' 

  return
end subroutine integrate_io_rk

subroutine update_perturber_positions( numparts, rparts)
  use orbit_integrator
  implicit none
  integer, intent(in) :: numparts
  double precision, dimension(numparts, 6), intent(in) :: rparts
  integer :: i
  double precision :: r, theta, z
  double precision :: vr, j, vz, vtheta

  do i = 1, numPerturbers

     r = rparts(i,1)
     vr = rparts(i,2)
     j = rparts(i,3)
     theta = rparts(i,4)
     z = rparts(i,5)
     vz = rparts(i,6)

     vtheta = j/(r*r)

     posp(i,1) = r*cos(theta)
     posp(i,2) = r*sin(theta)
     posp(i,3) = z
     
     vposp(i,1) = vr*cos(theta) - vtheta*r*sin(theta)
     vposp(i,2) = vr*sin(theta) + vtheta*r*cos(theta)
     vposp(i,3) = vz 
     
  end do

!  print *,'in update_perturber_positions'

  return
end subroutine update_perturber_positions


!  call stellar_cusp(r, acc, testenergy, massenc)


subroutine cross_product(a, b, c)
  implicit none
  
  double precision, dimension(3) :: a, b, c

  c(1) = a(2)*b(3) - b(2)*a(3)
  c(2) = a(3)*b(1) - b(3)*a(1)
  c(3) = a(1)*b(2) - b(1)*a(2)

  return
end subroutine cross_product

double precision function dot_product( a, b) result(ans)
  implicit none
  double precision, dimension(3), intent(in) :: a, b
  integer :: i
!  double precision :: ans

  ans = a(1)*b(1) + a(2)*b(2) + a(3)*b(3)
  return

end function dot_product

double precision function concMaccio(massp) result (conc)
  implicit none
  double precision :: massp
! Maccio et al. 2008, eqn 9 -- 

  conc = 0.971D0 - 0.094D0*log(massp/1.d12)
  conc = 10D0**conc

return

end function concMaccio

subroutine getprofile(r, mass, rho, sigma)
  use orbit_integrator
  implicit none

  double precision, intent(IN) :: r
  double precision, intent(OUT) :: mass, rho, sigma

  double precision :: sismass, rhosis, sigmasis, nfwmass, rhoNFW, sigmaNFW, &
       hernquistmass,rhohernquist,sigmahernquist
  external :: sismass, rhosis, sigmasis, nfwmass, rhoNFW, sigmaNFW, &
       hernquistmass,rhohernquist,sigmahernquist

   if(PROFILE_TYPE == 1) then
     mass = masssis
     rho = (V0sis**2)/(4.*pi*Gn*(r**2))
     sigma = V0sis/sqrt(2.)

     else if(PROFILE_TYPE == 2) then
     ! for NFW                                                                         
     ! get the NFW mass                                                                
     mass0 = Mvir/(log(1D0 + cvir) - cvir/(1D0+cvir))

     anfw = Rvir/cvir
     mass = nfwmass(anfw, mass0, r)
     rho = rhoNFW(anfw,Mvir,Rvir,cvir,r)
     sigma = sigmaNFW(r, Rvir, cvir, Mvir)

     else if(PROFILE_TYPE ==3) then
      ! hernquist profile w. same mass w.in R200 as NFW and same innner profile (Springel et al. 2005):
     mass = hernquistmass(r, mass0, v0200, c0)
     rho = rhohernquist(r,mass0,v0200,c0)
     sigma = sigmahernquist(r,mass0,v0200,c0)

  endif

  return
end subroutine getprofile

double precision function nfwmass(scalelength, mass,r) result(m) 
  use orbit_integrator
  implicit none
  
  double precision :: scalelength, mass, r
  double precision :: x,xRv

  x = r/scalelength
  xRv = Rvir/scalelength

  mass0 = Mvir/(log(1D0 + cvir) - cvir/(1D0+cvir))

  if (r.le.Rvir) then 
     m = mass0*(log(1+x) - (x/(1.+x)))
  else
     m=(mass0*(log(1+xRv) - (xRv/(1.+xRv))))
  endif 

  return
end function nfwmass

double precision function rhoNFW (scalelength,Mvir,Rvir,cvir,r) result(rho)

  implicit none
  
  double precision, parameter :: pi = 0.5D0*6.28318530717959d0
  double precision :: scalelength,Mvir,Rvir,cvir,r,x,fcvir,rhos

  fcvir = log(1.+cvir)-(cvir/(1.+cvir))
  rhos = Mvir/(4*pi*(scalelength**3)*fcvir)
  x = r/scalelength

  if (r.le.Rvir) then
     rho = rhos/(x*((1.+x)**2))
  else
     rho = 0.d0
  endif

  return
end function rhoNFW

subroutine derivs( noeqs, x, y, dydx)
  use orbit_integrator
  implicit none
  
  integer, intent(IN) :: noeqs
  double precision, intent(IN) :: x
  
  double precision, intent(IN), dimension(noeqs) :: y
  double precision, intent(OUT), dimension(noeqs) :: dydx
  double precision :: absr, absdeltar, mass
  double precision :: dot_product,rhoh
  integer :: gc
  integer :: k
  double precision :: hernquistmass
  external :: hernquistmass
  external :: dot_product,getprofile
  
  double precision :: r, vr, angmom, theta, rp, thetap,vtheta
  double precision, dimension(3) :: pos, deltax, acc, vpos, accdf ! dynamical friction term: Fdf/msat
  double precision :: costheta, sintheta
  
  double complex :: amr, amphi
  double complex, parameter :: Iimag = (0D0,1D0)
  external :: potential_m
  double precision :: accr, accphi
  double precision :: z, vz, zp
  double precision :: errorfunction,term1
  double precision :: erf,xchandra,msat,Lambda,logLambda,vscalar, sigma !sigma is 1-d velocity dispersion of halo
  integer :: m
  integer :: i 
  double precision :: errfunction
  external :: errfunction

  r = y(1)
  vr = y(2)
  angmom = y(3)
  theta = y(4)

  gc = int(y(7))

  z = y(5)
  vz = y(6)

  absr = 0D0
  pos = 0D0

  costheta = cos(theta)
  sintheta = sin(theta)

  pos(1) = r*costheta
  pos(2) = r*sintheta
  pos(3) = z

  vtheta = angmom/(r*r)

  vpos(1) = vr*cos(theta) - vtheta*r*sin(theta)
  vpos(2) = vr*sin(theta) + vtheta*r*cos(theta)
  vpos(3) = vz

  vscalar = sqrt(vpos(1)*vpos(1) + vpos(2)*vpos(2) + vpos(3)*vpos(3))

  absr = sqrt(dot_product(pos,pos))
           
  call getprofile( absr, mass, rhoh, sigma)

  accdf = 0.d0 

  if (gc.le.numPerturbers) then

     msat = massp(gc)  ! assign mass to perturber

     call getprofile( absr, mass, rhoh, sigma)

     xchandra = vscalar/(sqrt(2.D0)*sigma)
     erf = errfunction(xchandra)

     Lambda = absr/(3.*1.6*kpc)    ! following Besla et al. 2007 here.  TODO: vary bmin as function of perturber mass.
     
     logLambda=max(0.D0,log(Lambda))  ! to avoid problem when r < Rsat

     accdf = -4*pi*Gn**2*(logLambda)*rhoh*(1./vscalar**3)*vpos*&
      (erf-2*xchandra*(1./sqrt(pi))*(exp(-(xchandra**2))))*msat

  end if  ! ends dynamical friction consideration - add on below if needed.

  if (gc.le.numPerturbers) then
     acc = -(Gn*mass)*pos/(absr*absr*absr) + accdf
  else 
     acc = -(Gn*mass)*pos/(absr*absr*absr)  ! no dynamical friction from the perturber itself
  end if
  
  !    print *,'acc=',acc
  
  if(gc > 0D0) then
     
     do i = 1, numPerturbers
        if( gc /= i) then ! don't calculate self force
           deltax(:) = pos(:) - posp(i,:)
           absdeltar = sqrt(dot_product(deltax,deltax))
	   mass = hernquistmass( absdeltar, massp(i), vp200(i), cp(i))   ! because the perturber has *finite* size - need mass enclosed
	   
           acc = acc-(Gn*mass)*deltax/(absdeltar*absdeltar*absdeltar)
        end if
     end do
     accr = 0D0
     accphi = 0D0
!!$     do m = 1, 10
!!$        call potential_m(r, posp, m-1, amr, amphi)
!!$        accr = accr + 2D0*real(amr*exp(-Iimag*(m-1)*theta))
!!$        accphi = accphi + 2D0*real(amphi*exp(-Iimag*(m-1)*theta))
!!$     end do

  end if

  dydx = 0D0
!  acc = 0D0

  dydx(1) = vr
  dydx(2) = angmom*angmom/(r*r*r) + (acc(1)*costheta + acc(2)*sintheta) + accr
  dydx(3) = -r*(acc(1)*sintheta - acc(2)*costheta) + accphi
  dydx(4) = angmom/(r*r)
  dydx(5) = vz
  dydx(6) = acc(3)


  return
end subroutine derivs

SUBROUTINE rkckstep(noeqs, x1, x2, y, height, yout, yerr,derivatives)
  IMPLICIT NONE
  integer :: noeqs
  double precision, DIMENSION(noeqs), INTENT(IN) :: y
  double precision, INTENT(IN) :: x1, x2
  double precision :: x,h, test
  double precision, DIMENSION(noeqs), INTENT(OUT) :: yout,yerr
  double precision, dimension(noeqs), intent(OUT) :: height
  double precision, dimension(noeqs) :: dydx
  INTEGER :: ndum, i
  double precision, PARAMETER :: TINY=1.0D-30, HUGE = 1.0D30
  
  double precision, DIMENSION(noeqs) :: ak2,ak3,ak4,ak5,ak6,ytemp
  double precision, PARAMETER :: A2=0.2D0,A3=0.3D0,A4=0.6D0,A5=1.0D0,&
       A6=0.875D0,B21=0.2D0,B31=3.0D0/40.0D0,B32=9.0D0/40.0D0,&
       B41=0.3D0,B42=-0.9D0,B43=1.2D0,B51=-11.0D0/54.0D0,&
       B52=2.5D0,B53=-70.0D0/27.0D0,B54=35.0D0/27.0D0,&
       B61=1631.0D0/55296.0D0,B62=175.0D0/512.0D0,&
       B63=575.0D0/13824.0D0,B64=44275.0D0/110592.0D0,&
       B65=253.0D0/4096.0D0,C1=37.0D0/378.0D0,&
       C3=250.0D0/621.0D0,C4=125.0D0/594.0D0,&
       C6=512.0D0/1771.0D0,DC1=C1-2825.0D0/27648.0D0,&
       DC3=C3-18575.0D0/48384.0D0,DC4=C4-13525.0D0/55296.0D0,&
       DC5=-277.0D0/14336.0D0,DC6=C6-0.25D0
  
  external :: derivatives
  
  h = x2 - x1
  x = x1
  
  call derivatives(noeqs, x, y, dydx)    
  ytemp=y+B21*h*dydx
  call derivatives(noeqs, x+A2*h,ytemp,ak2)
  ytemp=y+h*(B31*dydx+B32*ak2)
  call derivatives(noeqs, x+A3*h,ytemp,ak3)
  ytemp=y+h*(B41*dydx+B42*ak2+B43*ak3)
  call derivatives(noeqs, x+A4*h,ytemp,ak4)
  ytemp=y+h*(B51*dydx+B52*ak2+B53*ak3+B54*ak4)
  call derivatives(noeqs, x+A5*h,ytemp,ak5)
  ytemp=y+h*(B61*dydx+B62*ak2+B63*ak3+B64*ak4+B65*ak5)
  call derivatives(noeqs, x+A6*h,ytemp,ak6)
  yout=y+h*(C1*dydx+C3*ak3+C4*ak4+C6*ak6)
  yerr=h*(DC1*dydx+DC3*ak3+DC4*ak4+DC5*ak5+DC6*ak6)
  
  ytemp = y + h*dydx + TINY

  yerr = yerr/ytemp
  
  do i = 1, noeqs
     test = abs(C1*dydx(i)+C3*ak3(i)+C4*ak4(i)+C6*ak6(i)) 
     if (test > 0D0) then
        height(i) = yout(i)/(C1*dydx(i)+C3*ak3(i)+C4*ak4(i)+C6*ak6(i)) 
     else
        height(i) = HUGE

     end if
  end do
END SUBROUTINE rkckstep


function min_step( height) result (min_height)
  implicit none
  integer, parameter :: noeqs = 4

  double precision, dimension (noeqs), intent (IN) :: height
  double precision :: min_height
  integer :: i
  
  min_height = abs(height(1))
  
  do i = 2, noeqs
     if(min_height > abs(height(i)) ) min_height = abs(height(i))
  end do
  
  return
  
end function min_step

function myfunc(xchandra) result (integrand)

  implicit none
  double precision, parameter :: pi = 0.5D0*6.28318530717959d0
  real, intent(in) :: xchandra
  double precision :: integrand
  integrand = (2./(sqrt(pi)))*(exp(-(xchandra**2)))  ! for error function in dyn friction term
  return

end function myfunc

function errfunction(xchandra) result (erf)
  use errorFunctionStuff

  implicit none
  double precision, intent(IN) :: xchandra
  double precision :: erf, derf
  double precision :: z, dz, dzmax, dzinit, dzmin
  double precision, dimension(1) :: y, yout, h, yerr
  external :: erfderivs, rkckstep
  double precision, parameter :: pi = 0.5D0*6.28318530717959d0
  integer :: i

  if( xchandra > ERROR_HI) then
     erf = 1D0
     return
  end if

  dz = ERROR_HI/(NUM_INTERVALS-1)

if(.not. errorFunctionCacheInitialized) then
     ! initialize the cache
     z = 0D0
     erf = 0D0
     y = 0D0
     errorFunctionCache = 0D0
     do i = 1, NUM_INTERVALS-1
        call rkckstep(1, dz*(i-1), dz*i, y, h, yout, yerr,erfderivs)
        y = yout
        errorFunctionCache(i+1) = 2D0/sqrt(pi)*y(1)
     end do
     errorFunctionCacheInitialized = .true.
  endif 
  
  ! find the value in the cache
  i = int(floor(xchandra*NUM_INTERVALS/ERROR_HI))
  derf = (errorFunctionCache(i+1) - errorFunctionCache(i))/dz
  
  ! find it via linear interpolation
  erf = errorFunctionCache(i) + derf*(xchandra-dz*i)

!  write(*,*) erf
!  pause
  return
end function errfunction

subroutine erfderivs( noeqs, x, y, dydx)
  implicit none
  integer, intent(IN) :: noeqs
  double precision, intent(IN) :: x
  double precision, intent(IN), dimension(noeqs) :: y
  double precision, intent(OUT), dimension(noeqs) :: dydx

  dydx(1) = exp(-x*x) 

  return
end subroutine erfderivs

double precision function sigmahernquist(r,m0,vc200,conc) result (sigma)

   implicit none
   double precision, intent (IN) :: r,m0,vc200,conc
   double precision, parameter :: Gn=6.7D-8
   double precision :: r200, a, rs,mass, Vmax, x
   double precision, parameter :: pi = 0.5D0*6.28318530717959d0
   double precision, parameter :: kpc = 3D21, kms=1D5

   Vmax = 215.D0*kms  ! for V200 = 160                                                                                 
   r200 = Gn*m0/(vc200*vc200)
   rs = r200/conc
   x = r/rs  ! Zentner & Bullock 2003, equation 6                                                                      
   sigma = Vmax*((1.439*(x**0.354))/(1.+1.1756*(x**0.725)))

return

end function sigmahernquist

double precision function sigmaNFW(r,Rvir,cvir,Mvir) result (sigma)

   implicit none
   double precision, intent (IN) :: r,Rvir,cvir,Mvir
   double precision, parameter :: Gn=6.7D-8
   double precision :: rs,Vmax, x, Vmaxsq, gcvir, Vvir
   double precision, parameter :: pi = 0.5D0*6.28318530717959d0
   double precision, parameter :: kpc = 3D21, kms=1D5


   ! following Zentner & Bullock 03, section 2.2, eqn 3                                                                      

   Vvir = sqrt(Gn*Mvir/Rvir)

   gcvir = log(1.+cvir)-(cvir/(1.+cvir))
   Vmaxsq = 0.216*(Vvir**2)*(cvir/gcvir)
   Vmax = sqrt(Vmaxsq)

   rs = Rvir/cvir
   x = r/rs  ! Zentner & Bullock 2003, equation 6
   
      sigma = Vmax*((1.439*(x**0.354))/(1.+1.1756*(x**0.725)))

return

end function sigmaNFW

double precision function rhohernquist(r,m0,vc200,conc) result (rho)

   implicit none
   double precision, intent (IN) :: r,m0,vc200,conc
   double precision, parameter :: Gn=6.7D-8
   double precision :: r200, a, rs,mass
   double precision, parameter :: pi = 0.5D0*6.28318530717959d0

   r200 = Gn*m0/(vc200*vc200)
   rs = r200/conc
   a = rs*sqrt(2D0*(log(1D0+conc) - conc/(1D0+conc)))
   mass = m0 * r*r/(r+a)**2D0
! m0 is M_dm, i.e., M(<r) = M_dm r^2/(r+a)^2   

   rho = (m0/(2.*pi))*(a/(r*(r+a)**3D0))
   return

end function rhohernquist

double precision function hernquistmass( r, m0, vc200, conc) result (mass)
  implicit none
  double precision, intent(IN) :: r, m0, vc200, conc
  double precision, parameter :: Gn=6.7D-8
  double precision :: r200, a, rs

  r200 = Gn*m0/(vc200*vc200)
  rs = r200/conc
  a = rs*sqrt(2D0*(log(1D0+conc) - conc/(1D0+conc)))

  mass = m0 * r*r/(r+a)**2D0

!  write (*,*) 'R200, Rs, a, m0, mass(at r=160 kpc)'
!  write (*,*) r200/3D21, rs/3D21, a/3D21, m0/1.9891D33, mass/1.9891D33
!  write (*,*) 'r='
!  write (*,*) r/3D21
 
  return
end function hernquistmass



!
! FFT 
!
SUBROUTINE dfour1(data,nn,isign)
  INTEGER :: isign,nn
  DOUBLE PRECISION :: data(2*nn)
  INTEGER :: i,istep,j,m,mmax,n
  DOUBLE PRECISION :: tempi,tempr
  DOUBLE PRECISION :: theta,wi,wpi,wpr,wr,wtemp
  n=2*nn
  j=1
  do i=1,n,2
     if(j.gt.i)then
        tempr=data(j)
        tempi=data(j+1)
        data(j)=data(i)
        data(j+1)=data(i+1)
        data(i)=tempr
        data(i+1)=tempi
     endif
     m=n/2
     do while ((m.ge.2).and.(j.gt.m))
        j=j-m
        m=m/2
     end do
     j=j+m
  enddo
  mmax=2
  do while (n.gt.mmax)
     istep=2*mmax
     theta=6.28318530717959d0/(isign*mmax)
     wpr=-2.d0*sin(0.5d0*theta)**2
     wpi=sin(theta)
     wr=1.d0
     wi=0.d0
     do m=1,mmax,2
        do i=m,n,istep
           j=i+mmax
           tempr=wr*data(j)-wi*data(j+1)
           tempi=wr*data(j+1)+wi*data(j)
           data(j)=data(i)-tempr
           data(j+1)=data(i+1)-tempi
           data(i)=data(i)+tempr
           data(i+1)=data(i+1)+tempi
        end do
        wtemp=wr
        wr=wr*wpr-wi*wpi+wr
        wi=wi*wpr+wtemp*wpi+wi
     end do
     mmax=istep
  end do
  return
END SUBROUTINE dfour1


subroutine writedata(tau, numPerturbers, numparts, rparts, cnt)
  implicit none
  integer :: numparts, i, numPerturbers, cnt
  double precision, parameter :: kms=1D5
  double precision :: tau
  double precision, dimension(numparts,6) :: rparts, vparts
  double precision :: x,y,r,theta, z, vx, vy, vz, vtheta, j, vr
  double precision :: absl, absr, v2, energy
  double precision, dimension(numparts) :: eps, semi
  character(LEN=40) :: filename
!  write( filename, fmt='(A16,E10.3,A8)') "particle_data_t=", tau, "Gyr.data"
  write( filename, fmt='(A13,I3,A4)') "particle_data",cnt,".txt"
  open(unit=22, file=filename, status='new')

  write(22,fmt='(F5.3)') tau

!  do i = numPerturbers, numparts, 1
   do i=1,numparts
     r = rparts(i,1)
     vr = rparts(i,2)
     j = rparts(i,3)
     theta = rparts(i,4)

     vtheta = j/(r*r)

     x = r*cos(theta)
     y = r*sin(theta)
     z = rparts(i,5)

     vx = vr*cos(theta) - vtheta*r*sin(theta)
     vy = vr*sin(theta) + vtheta*r*cos(theta)
     vz = rparts(i,6)
     write(22,fmt='(I5,9(1X,E12.4))') i, x, y, z, vx/kms, vy/kms, vz/kms, r, theta

!  write(*,fmt='(11(2X,D12.6))') tau, rparts(1,1), rparts(1,2), rparts(2,1), rparts(2,2)
     
  end do
  close(22) 
!  write(24)tau,rparts,vparts

  return 
end subroutine writedata



