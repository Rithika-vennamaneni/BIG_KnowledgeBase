"""
Azure OCR Processor for Healthcare Payer Rule Extraction
Reads PDFs from Azure Blob Storage and saves JSON results back to Azure
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class HealthcareOCRProcessor:
    """
    OCR processor for healthcare payer documents
    Reads PDFs from Azure Blob Storage, processes with OCR, saves JSON to Azure
    """
    
    def __init__(self):
        """Initialize Azure Document Intelligence and Storage clients"""
        # Document Intelligence setup
        doc_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        doc_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        
        if not doc_endpoint or not doc_key:
            raise ValueError(
                "Missing Azure Document Intelligence credentials:\n"
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT\n"
                "AZURE_DOCUMENT_INTELLIGENCE_KEY"
            )
        
        self.doc_client = DocumentIntelligenceClient(
            endpoint=doc_endpoint,
            credential=AzureKeyCredential(doc_key)
        )
        print("✓ Azure Document Intelligence client initialized")
        
        # Azure Storage setup
        storage_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not storage_conn:
            raise ValueError("Missing AZURE_STORAGE_CONNECTION_STRING in .env file")
        
        self.blob_service = BlobServiceClient.from_connection_string(storage_conn)
        self.pdf_container = os.getenv("AZURE_PDF_CONTAINER_NAME", "healthcare-pdfs")
        self.json_container = os.getenv("AZURE_JSON_CONTAINER_NAME", "healthcare-json")
        
        # Create JSON container if it doesn't exist
        try:
            self.blob_service.create_container(self.json_container)
            print(f"✓ Created new container: {self.json_container}")
        except Exception:
            print(f"✓ Using existing container: {self.json_container}")
        
        print(f"✓ Azure Storage configured")
        print(f"  - PDF Container: {self.pdf_container}")
        print(f"  - JSON Container: {self.json_container}")
    
    def generate_blob_sas_url(self, container_name: str, blob_name: str) -> str:
        """Generate a SAS URL for a blob (valid for 2 hours)"""
        sas_token = generate_blob_sas(
            account_name=self.blob_service.account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=self.blob_service.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=2)
        )
        
        blob_url = f"https://{self.blob_service.account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
        return blob_url
    
    def process_pdf_from_blob(self, blob_name: str) -> Dict:
        """
        Process a PDF from Azure Blob Storage with OCR
        
        Args:
            blob_name: Name of the PDF blob in the container
            
        Returns:
            Dictionary containing extracted healthcare content
        """
        print(f"\n{'='*70}")
        print(f"Processing: {blob_name}")
        print(f"{'='*70}")
        
        try:
            # Generate SAS URL for the PDF
            pdf_url = self.generate_blob_sas_url(self.pdf_container, blob_name)
            
            # Start OCR analysis
            print("⏳ Sending PDF to Azure Document Intelligence...")
            poller = self.doc_client.begin_analyze_document_from_url(
                model_id="prebuilt-layout",
                document_url=pdf_url
            )
            
            print("⏳ Analyzing document (this may take 10-60 seconds)...")
            result = poller.result()
            
            # Convert to healthcare-specific JSON format
            print("⏳ Extracting healthcare rules and content...")
            json_result = self._convert_to_healthcare_json(result, blob_name)
            
            # Save JSON to Azure Blob Storage
            json_blob_name = Path(blob_name).stem + "_ocr.json"
            self._upload_json_to_blob(json_result, json_blob_name)
            
            # Print summary
            self._print_extraction_summary(json_result)
            
            return json_result
            
        except Exception as e:
            print(f"✗ Error processing {blob_name}: {str(e)}")
            raise
    
    def process_all_pdfs_from_blob(self) -> List[Dict]:
        """
        Process all PDFs from Azure Blob Storage container
        
        Returns:
            List of processing results
        """
        print(f"\n{'='*70}")
        print(f"Scanning PDF container: {self.pdf_container}")
        print(f"{'='*70}")
        
        # Get container client
        container_client = self.blob_service.get_container_client(self.pdf_container)
        
        # List all PDF blobs
        try:
            all_blobs = list(container_client.list_blobs())
            pdf_blobs = [blob for blob in all_blobs if blob.name.lower().endswith('.pdf')]
        except Exception as e:
            print(f"✗ Error accessing container: {str(e)}")
            print("Make sure the container exists and you have permissions")
            return []
        
        if not pdf_blobs:
            print(f"⚠️  No PDF files found in container '{self.pdf_container}'")
            return []
        
        print(f"✓ Found {len(pdf_blobs)} PDF files to process")
        
        # Estimate cost
        total_size_mb = sum(blob.size for blob in pdf_blobs) / (1024 * 1024)
        estimated_pages = len(pdf_blobs) * 15  # Conservative estimate
        estimated_cost = (estimated_pages / 1000) * 1.50
        
        print(f"\n📊 Processing Statistics:")
        print(f"   Total PDFs: {len(pdf_blobs)}")
        print(f"   Total size: {total_size_mb:.2f} MB")
        print(f"   Estimated pages: {estimated_pages}")
        print(f"   Estimated cost (S0 tier): ${estimated_cost:.2f}")
        print(f"   (Free F0 tier: 500 pages/month included)")
        
        # Process each PDF
        results = []
        for i, blob in enumerate(pdf_blobs, 1):
            print(f"\n{'─'*70}")
            print(f"[{i}/{len(pdf_blobs)}] Processing: {blob.name}")
            print(f"{'─'*70}")
            
            try:
                json_result = self.process_pdf_from_blob(blob.name)
                results.append({
                    "blob_name": blob.name,
                    "status": "success",
                    "pages": json_result.get("page_count", 0),
                    "rules_extracted": json_result.get("healthcare_rules_count", 0),
                    "json_blob": Path(blob.name).stem + "_ocr.json",
                    "timestamp": datetime.utcnow().isoformat()
                })
                print(f"✓ Success")
            except Exception as e:
                print(f"✗ Failed: {str(e)}")
                results.append({
                    "blob_name": blob.name,
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Rate limiting to avoid throttling
            if i < len(pdf_blobs):
                time.sleep(3)  # 3 second delay between requests
        
        # Upload batch summary to Azure
        self._upload_batch_summary(results)
        
        # Print final summary
        self._print_batch_summary(results)
        
        return results
    
    def _upload_json_to_blob(self, json_data: Dict, blob_name: str):
        """Upload JSON data to Azure Blob Storage"""
        try:
            blob_client = self.blob_service.get_blob_client(
                container=self.json_container,
                blob=blob_name
            )
            
            json_string = json.dumps(json_data, indent=2, ensure_ascii=False)
            blob_client.upload_blob(
                json_string,
                overwrite=True,
                content_settings={'content_type': 'application/json'}
            )
            
            print(f"✓ JSON uploaded to: {self.json_container}/{blob_name}")
            
        except Exception as e:
            print(f"✗ Error uploading JSON: {str(e)}")
            raise
    
    def _upload_batch_summary(self, results: List[Dict]):
        """Upload batch processing summary to Azure"""
        summary = {
            "processing_date": datetime.utcnow().isoformat(),
            "total_files": len(results),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "failed"]),
            "total_pages": sum(r.get("pages", 0) for r in results if r["status"] == "success"),
            "total_rules": sum(r.get("rules_extracted", 0) for r in results if r["status"] == "success"),
            "results": results
        }
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        summary_blob_name = f"_batch_summary_{timestamp}.json"
        
        try:
            self._upload_json_to_blob(summary, summary_blob_name)
            print(f"\n✓ Batch summary uploaded: {summary_blob_name}")
        except Exception as e:
            print(f"\n⚠️  Could not upload summary: {str(e)}")
    
    def _convert_to_healthcare_json(self, result, blob_name: str) -> Dict:
        """Convert Azure OCR result to healthcare-specific JSON format"""
        json_output = {
            "source_blob": blob_name,
            "extraction_date": datetime.utcnow().isoformat(),
            "page_count": len(result.pages) if result.pages else 0,
            "full_text": result.content,
            
            # Structured content
            "pages": [],
            "paragraphs": [],
            "tables": [],
            
            # Healthcare-specific metadata
            "healthcare_metadata": {
                "payer_name": self._extract_payer_name(result.content),
                "document_type": self._classify_document_type(result.content),
                "detected_rules": self._extract_healthcare_rules(result.content)
            },
            
            "healthcare_rules_count": 0
        }
        
        # Extract page information
        if result.pages:
            for page in result.pages:
                page_data = {
                    "page_number": page.page_number,
                    "dimensions": {
                        "width": page.width,
                        "height": page.height,
                        "unit": page.unit
                    },
                    "lines": []
                }
                
                if page.lines:
                    for line in page.lines:
                        page_data["lines"].append({
                            "text": line.content,
                            "confidence": getattr(line, 'confidence', None),
                            "bounding_box": line.polygon if hasattr(line, 'polygon') else None
                        })
                
                json_output["pages"].append(page_data)
        
        # Extract paragraphs
        if result.paragraphs:
            for para in result.paragraphs:
                para_data = {
                    "content": para.content,
                    "role": getattr(para, 'role', None),
                }
                
                if hasattr(para, 'bounding_regions') and para.bounding_regions:
                    para_data["page_number"] = para.bounding_regions[0].page_number
                
                json_output["paragraphs"].append(para_data)
        
        # Extract tables
        if result.tables:
            for table_idx, table in enumerate(result.tables):
                table_data = {
                    "table_id": table_idx + 1,
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                    "cells": []
                }
                
                for cell in table.cells:
                    table_data["cells"].append({
                        "row": cell.row_index,
                        "column": cell.column_index,
                        "content": cell.content,
                        "kind": getattr(cell, 'kind', None),
                        "row_span": getattr(cell, 'row_span', 1),
                        "column_span": getattr(cell, 'column_span', 1)
                    })
                
                json_output["tables"].append(table_data)
        
        # Update healthcare rules count
        json_output["healthcare_rules_count"] = len(
            json_output["healthcare_metadata"]["detected_rules"]
        )
        
        return json_output
    
    def _extract_payer_name(self, text: str) -> Optional[str]:
        """Extract payer name from document"""
        payer_keywords = [
            "united healthcare", "unitedhealthcare", "uhc",
            "anthem", "elevance",
            "aetna", "cvs health",
            "kaiser permanente",
            "cigna", "humana",
            "blue cross blue shield", "bcbs",
            "centene", "molina healthcare"
        ]
        
        text_lower = text.lower()
        for payer in payer_keywords:
            if payer in text_lower:
                return payer.title()
        return None
    
    def _classify_document_type(self, text: str) -> str:
        """Classify healthcare document type"""
        text_lower = text.lower()
        
        if "prior authorization" in text_lower or "pre-authorization" in text_lower:
            return "Prior Authorization Guidelines"
        elif "claim" in text_lower and "timely filing" in text_lower:
            return "Claims & Timely Filing"
        elif "appeal" in text_lower:
            return "Appeals Process"
        elif "provider manual" in text_lower:
            return "Provider Manual"
        elif "fee schedule" in text_lower or "reimbursement" in text_lower:
            return "Fee Schedule"
        elif "credentialing" in text_lower:
            return "Credentialing Requirements"
        else:
            return "General Policy Document"
    
    def _extract_healthcare_rules(self, text: str) -> List[Dict]:
        """Extract healthcare rules using pattern matching"""
        rules = []
        text_lower = text.lower()
        
        rule_patterns = {
            "prior_authorization": ["prior authorization required", "pre-auth required", "pa required"],
            "timely_filing": ["timely filing", "filing deadline", "claim submission deadline"],
            "appeal_deadline": ["appeal within", "appeal deadline", "days to appeal"],
            "documentation": ["documentation required", "medical records required"],
            "notification": ["notification required", "must notify", "advance notice"]
        }
        
        for rule_type, patterns in rule_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    rules.append({
                        "type": rule_type,
                        "pattern_matched": pattern,
                        "rule_text": self._extract_surrounding_text(text, pattern)
                    })
        
        return rules
    
    def _extract_surrounding_text(self, text: str, pattern: str, context_chars: int = 200) -> str:
        """Extract text around a matched pattern"""
        pattern_lower = pattern.lower()
        text_lower = text.lower()
        
        idx = text_lower.find(pattern_lower)
        if idx == -1:
            return ""
        
        start = max(0, idx - context_chars)
        end = min(len(text), idx + len(pattern) + context_chars)
        
        return text[start:end].strip()
    
    def _print_extraction_summary(self, json_result: Dict):
        """Print extraction summary"""
        print(f"\n{'─'*70}")
        print("EXTRACTION SUMMARY")
        print(f"{'─'*70}")
        print(f"Pages: {json_result['page_count']}")
        print(f"Paragraphs: {len(json_result['paragraphs'])}")
        print(f"Tables: {len(json_result['tables'])}")
        print(f"Healthcare Rules: {json_result['healthcare_rules_count']}")
        
        metadata = json_result['healthcare_metadata']
        if metadata['payer_name']:
            print(f"Payer: {metadata['payer_name']}")
        print(f"Document Type: {metadata['document_type']}")
        
        if metadata['detected_rules']:
            print(f"\nRule Types Found:")
            rule_types = set(rule['type'] for rule in metadata['detected_rules'])
            for rule_type in rule_types:
                count = len([r for r in metadata['detected_rules'] if r['type'] == rule_type])
                print(f"  • {rule_type.replace('_', ' ').title()}: {count}")
    
    def _print_batch_summary(self, results: List[Dict]):
        """Print batch processing summary"""
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]
        
        print(f"\n{'='*70}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"\n✅ Successfully processed: {len(successful)} files")
        print(f"❌ Failed: {len(failed)} files")
        
        if successful:
            total_pages = sum(r.get("pages", 0) for r in successful)
            total_rules = sum(r.get("rules_extracted", 0) for r in successful)
            print(f"\n📄 Total pages extracted: {total_pages}")
            print(f"📋 Total healthcare rules found: {total_rules}")
            print(f"💾 JSON files saved to container: {self.json_container}")
        
        if failed:
            print(f"\n❌ Failed files:")
            for result in failed[:5]:  # Show first 5 failures
                print(f"   • {result['blob_name']}: {result.get('error', 'Unknown error')}")
            if len(failed) > 5:
                print(f"   ... and {len(failed) - 5} more")


# Example usage
if __name__ == "__main__":
    processor = HealthcareOCRProcessor()
    
    # Process all PDFs from Azure Blob Storage
    results = processor.process_all_pdfs_from_blob()