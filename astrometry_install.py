# If you have Windows, run this in PowerShell or Command Prompt to install Ubuntu:
wsl --install

#----------------------------------------------------------------------------------------------------
# Step 1: In Ubuntu, run these commands to install the astrometry engine:
sudo apt update # Will ask for your Ubuntu password
sudo apt install astrometry.net netpbm libnetpbm-dev -y

#----------------------------------------------------------------------------------------------------
# OPTIONAL STEP A: Set up D drive storage (Highly recommended to save space on C:)
# Skip this section if you want to install everything on your default Linux drive.
mkdir -p /mnt/d/AstrometryIndex
cd /mnt/d/AstrometryIndex

# Delete the default system directory so we can replace it:
sudo rm -rf /usr/share/astrometry
# Create a symbolic link (shortcut) pointing to your D: drive folder:
sudo ln -s /mnt/d/AstrometryIndex /usr/share/astrometry
#----------------------------------------------------------------------------------------------------

# Step 2: Download the Index Files
# Note: If you used Optional Step A above, you are already in your D drive folder and can just use 'wget'.
# If you skipped the optional step, navigate to the default folder below. Because it is a system folder, 
# you must add 'sudo' (Superuser Do) before the download commands.

# run this to navigate to the default folder (Skip if you've ran the optional step A)
cd /usr/share/astrometry

# Download high-density narrow-field indices (Scale 5 and 6):
# Below are just some recomendations/examples. Go to https://data.astrometry.net/ and find the index you need.
# Again, you must add 'sudo' (Superuser Do) before the download commands if you skipped the optional steps.
for i in {00..05}; do
    wget https://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE/index-5206-$i.fits
    wget https://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE/index-5205-$i.fits
done

for scale in 5200 5201 5202 5203 5204; do
    for patch in {00..05}; do
        wget https://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE/index-${scale}-${patch}.fits
    done

for scale in {5200..5204}; do
    for patch in 12 13 14 15 26 27 28; do
        wget https://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE/index-${scale}-${patch}.fits
    done
done

#----------------------------------------------------------------------------------------------------
# Step 3: Verify It Worked
# List the files inside the configuration directory to ensure the engine sees them:
ls /usr/share/astrometry
# You should see a list of your downloaded .fits index files.
#----------------------------------------------------------------------------------------------------