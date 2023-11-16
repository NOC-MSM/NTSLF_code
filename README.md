# NTSLF_code
(Some) code for the NTSLF website

The idea is that this would be a good place to keep code for the autogeneration of NTSLF plots.

At present (Jul'23) there are two scripts.
Upon completion of image file creation, both scripts sftp the SVG files to:

        livftp.noc.ac.uk/local/users/ntslf/pub/ntslf_surge_animation/

---

**Animations of sea level and surge**: (`surge_anim.py`)

Example frames:

![ssh_latest_0240](https://github.com/NOC-MSM/NTSLF_code/assets/22616872/7d785f34-f677-4383-b35f-97e6b68bda5b)
![surge_anom_latest_0240](https://github.com/NOC-MSM/NTSLF_code/assets/22616872/03b5e4f8-718c-43db-93b2-219d8813c615)

![Preview surge](https://noc-msm.github.io/NTSLF_code/index.html)

---

**Time series plots of 7-day surge forecast using the meteorological forcing ensemble**: (`surge_ens.py`)

(This is experimental)
  
![Liverpool surge forecast ensemble example](https://github.com/NOC-MSM/NTSLF_code/assets/22616872/5efd4422-1e82-438b-a4c0-e822572d9db1)
